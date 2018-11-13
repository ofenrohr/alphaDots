import sys
sys.path.append('..')
from protobuf.py import DotsAndBoxesImage_pb2 as pb
from protobuf.py import GameSequence_pb2 as ProtobufGameSequence
import numpy as np


class Sequence:
    def __init__(self, model):
        print("Sequence init")
        self.model = model

    def deserializeProtobuf(self, message):
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

        prediction = self.model.predict(npseq)
        prediction_img = np.array(prediction * 255, dtype=np.uint8)

        print("prediction:")
        print(prediction_img[0,-1,:,:,0])

        prediction_data_pb = pb.DotsAndBoxesImage()
        prediction_data_pb.width = seq.width
        prediction_data_pb.height = seq.height

        for y in range(seq.height):
            for x in range(seq.width):
                prediction_data_pb.pixels.append(prediction_img[0, -1, y, x, 0])

        return prediction_data_pb
