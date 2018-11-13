import sys
sys.path.append('..')
from protobuf.py import DotsAndBoxesImage_pb2 as pb
from protobuf.py import GameSequence_pb2 as ProtobufGameSequence
import numpy as np


class SequenceCategorical:
    def __init__(self, model):
        print("Sequence init")
        self.model = model
        for layer in self.model.layers:
            print(layer.input_shape)

    def imgSizeToBoxes(self, x):
        if x is None:
            return None
        return (x - 3) / 2

    def linesToDotsAndBoxesImage(self, lines, imgWidth, imgHeight):
        boxWidth = self.imgSizeToBoxes(imgWidth)
        boxHeight = self.imgSizeToBoxes(imgHeight)
        linesCnt = 2 * boxWidth * boxHeight + boxWidth + boxHeight
        mat = np.zeros((imgHeight, imgWidth), dtype=lines.dtype)
        for idx in range(linesCnt):
            y1 = idx / ((2 * boxWidth) + 1)
            if idx % ((2 * boxWidth) + 1) < boxWidth:
                # horizontal line
                x1 = idx % ((2 * boxWidth) + 1)
                x2 = x1 + 1
                y2 = y1
            else:
                # vertical line
                x1 = idx % ((2 * boxWidth) + 1) - boxWidth
                x2 = x1
                y2 = y1 + 1
            px = x2 * 2 + y2 - y1
            py = y2 * 2 + x2 - x1
            mat[py, px] = lines[idx]
        return mat

    def derserializeProtobuf(self, message):
        seq = ProtobufGameSequence.GameSequence()
        seq.ParseFromString(message)
        return seq

    def predict(self, seq):
        print("received message")
        sys.stdout.flush()

        print(str(seq.width) + " x " + str(seq.height))
        sys.stdout.flush()

        print("game length: " + str(len(seq.game)))
        npseq = np.zeros(shape=(1,len(seq.game), seq.height, seq.width, 1))
        for frame in range(len(seq.game)):
            npframe = np.array(seq.game[frame].input)
            npframe = npframe.reshape(seq.height, seq.width, 1)
            npseq[0,frame] = npframe / 255.0
            #print(npframe[:,:,0])
            #print("~"*80)
        print("input:")
        print(npseq[0,-1,:,:,0])

        prediction_categorical = self.model.predict(npseq)
        print("categorical prediction shape:")
        print(prediction_categorical.shape)
        prediction = self.linesToDotsAndBoxesImage(prediction_categorical[0,-1,:], seq.width, seq.height)
        prediction_img = np.array(prediction * 255, dtype=np.uint8)

        print("prediction:")
        print(prediction_img)

        prediction_data_pb = pb.DotsAndBoxesImage()
        prediction_data_pb.width = seq.width
        prediction_data_pb.height = seq.height

        for y in range(seq.height):
            for x in range(seq.width):
                prediction_data_pb.pixels.append(prediction_img[y, x])

        return prediction_data_pb
