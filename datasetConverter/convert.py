import numpy as np
from PIL import Image
import os
import sys
import re
import argparse
import zmq

import DotsAndBoxesImage_pb2 as pb
import TrainingExample_pb2
import GameSequence_pb2

parser = argparse.ArgumentParser(description='Process Dots and Boxes datasets')
parser.add_argument('--zmq', type=int)
parser.add_argument('--seq', action='store_true')
parser.add_argument('--seq2', action='store_true')
parser.add_argument('--dataset-dir', default='/run/media/ofenrohr/Data/AlphaDots/data/firstTry/')
parser.add_argument('--output-file', default='/run/media/ofenrohr/Data/AlphaDots/data/firstTry.npz')
parser.add_argument('--x-size', default=13, type=int)
parser.add_argument('--y-size', default=11, type=int)
parser.add_argument('--debug', action='store_true')


args = parser.parse_args()

if args.debug:
    print(args)
    sys.stdout.flush()

dataset_dir = args.dataset_dir
output_file = args.output_file
x_size = int(args.x_size)
y_size = int(args.y_size)
zmqarg = args.zmq
seqarg = args.seq
seq2arg = args.seq2
debugarg = args.debug

print("config:")
print("|-> dataset_dir = " + dataset_dir)
print("|-> output_file = " + output_file)
print("|-> x_size = " + str(x_size))
print("|-> y_size = " + str(y_size))
print("|-> zmq = " + str(zmqarg))
print("|-> seq = " + str(seqarg))
print("|-> seq2 = " + str(seq2arg))
sys.stdout.flush()


def checkInstanceGUID(a,b):
    m = re.search('\{([a-f\d]{8}(-[a-f\d]{4}){3}-[a-f\d]{12}?)\}',a)
    if m.group(1) not in b:
        print("ERROR GUIDs don't match! "+m.group(1))
        return False
    return True


def readImg(img_path):
    img = Image.open(img_path).convert("L")
    # img.size
    arr = np.array(img.getdata(), dtype=np.uint8)
    npimg = np.resize(arr, (img.size[1], img.size[0]))
    return npimg


def receiveImg(message):
    img = pb.DotsAndBoxesImage()
    img.ParseFromString(message)
    if x_size != img.width or y_size != img.height:
        print("board size invalid")
        print("expected: " + str(x_size) + "x" + str(y_size))
        print("received: " + str(img.width) + "x" + str(img.height))
        return None
    npimg = np.array(img.pixels)
    npimg = npimg.reshape(img.height, img.width)
    return npimg


def receiveTrainingExample(message):
    ex = TrainingExample_pb2.TrainingExample()
    ex.ParseFromString(message)
    if x_size != ex.width or y_size != ex.height:
        print("board size invalid")
        print("expected: " + str(x_size) + "x" + str(y_size))
        print("received: " + str(ex.width) + "x" + str(ex.height))
        return None
    inp = np.array(ex.input)
    inp = inp.reshape(ex.height, ex.width)
    outp = np.array(ex.output)
    outp = outp.reshape(ex.height, ex.width)

    return inp, outp


def receiveTrainingSequence(message):
    ex = GameSequence_pb2.GameSequence()
    ex.ParseFromString(message)
    if x_size != ex.width or y_size != ex.height:
        print("board size invalid (seq)")
        print("expected: " + str(x_size) + "x" + str(y_size))
        print("received: " + str(ex.width) + "x" + str(ex.height))
        return None
    bw = (ex.width - 3) / 2
    bh = (ex.height - 3) / 2
    seqlen = 2*bw*bh+bw+bh
    assert(seqlen == len(ex.game))
    seq2 = len(ex.game[0].output) > 0
    if not seq2:
        seqdata = np.zeros(shape=(seqlen, ex.height, ex.width), dtype=np.uint8)
        for i in range(seqlen):
            frame = np.array(ex.game[i].input)
            seqdata[i] = frame.reshape(ex.height, ex.width)
        #seqdata = seqdata.reshape(seqlen, ex.height, ex.width) # WARNING: this assumes full games

        return seqdata
    else:
        input_seq = np.zeros(shape=(seqlen, ex.height, ex.width), dtype=np.uint8)
        target_seq = np.zeros(shape=(seqlen, ex.height, ex.width), dtype=np.uint8)
        for i in range(seqlen):
            frame_in = np.array(ex.game[i].input)
            input_seq[i] = frame_in.reshape(ex.height, ex.width)
            frame_target = np.array(ex.game[i].output)
            target_seq[i] = frame_target.reshape(ex.height, ex.width)

        if debugarg:
            print(input_seq[-1])
            print("~"*40)
            print(target_seq[-1])
        return input_seq, target_seq


socket = None

if args.zmq is not None:
    if debugarg:
        print("0mq")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:12355")

    sample_cnt = int(args.zmq)
    if debugarg:
        print("sample_cnt = "+str(sample_cnt))

    sys.stdout.flush()

    if not seqarg and not seq2arg: #receive single training examples
        x_dataset = np.zeros(shape=(sample_cnt, y_size, x_size), dtype=np.uint8)
        y_dataset = np.zeros(shape=(sample_cnt, y_size, x_size), dtype=np.uint8)

        for i in range(sample_cnt):
            message = socket.recv()

            inp, outp = receiveTrainingExample(message)

            x_dataset[i] = inp
            y_dataset[i] = outp

            socket.send_string("ok")

            if debugarg:
                print(inp)
                print(outp)
                print("="*80)

    # receive GameSequences (only input -> input and target are the same)
    elif seqarg:
        bw = (x_size-3) / 2
        bh = (y_size-3) / 2
        training_seq = np.zeros(shape=(sample_cnt, bw*bh*2+bw+bh, y_size, x_size), dtype=np.uint8)
        #print("training sequence shape: " + str(training_seq.shape))
        #sys.stdout.flush()

        for i in range(sample_cnt):
            message = socket.recv()

            #print("got msg")
            #sys.stdout.flush()

            seqData = receiveTrainingSequence(message)

            training_seq[i] = seqData

            socket.send_string("ok")

            if debugarg:
                print(seqData)
                print("="*80)
                sys.stdout.flush()

    # receive GameSequences (input and target sequences)
    elif seq2arg:
        # convert image size to boxes size
        bw = (x_size-3) / 2
        bh = (y_size-3) / 2
        lines_cnt = bw*bh*2+bw+bh
        input_seqs = np.zeros(shape=(sample_cnt, lines_cnt, y_size, x_size), dtype=np.uint8)
        target_seqs = np.zeros(shape=(sample_cnt, lines_cnt, y_size, x_size), dtype=np.uint8)

        for i in range(sample_cnt):
            message = socket.recv()

            (input_seq, target_seq) = receiveTrainingSequence(message)
            input_seqs[i] = input_seq
            target_seqs[i] = target_seq

            socket.send_string("ok")

            if debugarg:
                print(input_seq[-10])
                print("~"*40)
                print(target_seq[-10])
                print("="*80)
                sys.stdout.flush()




# used for first try dataset which used individual files for each example
# not zmq:
else:
    file_list = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
    file_list = sorted(file_list)

    sample_cnt = len(file_list)/2
    print("sample_cnt = "+str(sample_cnt))

    x_dataset = np.zeros(shape=(sample_cnt, y_size, x_size), dtype=np.uint8)
    y_dataset = np.zeros(shape=(sample_cnt, y_size, x_size), dtype=np.uint8)

    for i in range(0, sample_cnt*2, 2):
        x_img_path = file_list[i]
        y_img_path = file_list[i+1]
        if not checkInstanceGUID(x_img_path, y_img_path):
            break
        x_img = readImg(x_img_path)
        y_img = readImg(y_img_path)
        if i == 0:
            print(x_img.shape)
            print(y_img.shape)
        x_dataset[i/2] = x_img
        y_dataset[i/2] = y_img
        if i%1000 == 0:
            print ".",

print("saving data...")

if not seqarg and not seq2arg:
    np.savez(output_file, x_train=x_dataset, y_train=y_dataset)
elif seqarg:
    np.savez(output_file, games=training_seq)
elif seq2arg:
    np.savez(output_file, input_seq=input_seqs, target_seq=target_seqs)

print("done saving data...")
