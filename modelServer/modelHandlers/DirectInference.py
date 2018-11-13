import sys
sys.path.append('..')
from protobuf.py import DotsAndBoxesImage_pb2 as pb
from protobuf.py import GameSequence_pb2 as ProtobufGameSequence
import numpy as np
import keras.backend as K
from PIL import Image

class DirectInference:
    def __init__(self, model, debug=False):
        self.model = model
        self.debug = debug

    def deserializeProtobuf(self, protobuf_message):
        img = pb.DotsAndBoxesImage()
        img.ParseFromString(protobuf_message)
        return img

    def predict(self, img):
        if self.debug:
            print("received message")
            sys.stdout.flush()

            print(str(img.width) + " x " + str(img.height))
            # print(img.pixels)
            sys.stdout.flush()

        npimg = np.array(img.pixels)
        npimg = npimg.reshape(1, img.height, img.width, 1)
        npimgCpy = npimg.copy()
        npimg = npimg.astype(K.floatx())
        npimg /= 255.0

        if self.debug:
            print("request: ")
            print(np.floor(npimg[0, :, :, 0] * 100).astype(np.uint8))
            sys.stdout.flush()

        prediction = self.model.predict(npimg)

        prediction_data_print = prediction[0, :, :, 1] * 100
        prediction_data_print = prediction_data_print.astype(np.uint8)
        if self.debug:
            print("prediction: ")
            print(prediction_data_print)
            sys.stdout.flush()

        prediction_data_img = prediction[0, :, :, 1] * 255
        prediction_data_img = prediction_data_img.astype(np.uint8)

        # create image
        if self.debug:
            # merge image data in color channels
            tmp = np.zeros((prediction[0].shape[0], prediction[0].shape[1]), dtype=np.uint8)
            merged_imgdata = np.stack([npimgCpy[0, :, :, 0].astype(np.uint8), prediction_data_img, tmp], axis=-1)

            imgDebug = Image.fromarray(merged_imgdata, 'RGB')
            imgDebug = imgDebug.resize(size=(imgDebug.size[0] * 10, imgDebug.size[1] * 10))
            imgDebug.save('/tmp/debug.png')
            # imgDebug.show()

        # send data
        prediction_data_pb = pb.DotsAndBoxesImage()
        prediction_data_pb.width = img.width
        prediction_data_pb.height = img.height

        for y in range(img.height):
            for x in range(img.width):
                prediction_data_pb.pixels.append(prediction_data_img[y, x])

        return prediction_data_pb
