import sys
sys.path.append('..')
from protobuf.py import DotsAndBoxesImage_pb2 as pb
from protobuf.py import GameSequence_pb2 as ProtobufGameSequence
import numpy as np
from keras.utils.np_utils import to_categorical
from PIL import Image

class DirectInferenceCategorical:
    def __init__(self, model):
        print("DirectInferenceCategorical")
        self.model = model

    def deserializeProtobuf(self, protobuf_message):
        img = pb.DotsAndBoxesImage()
        img.ParseFromString(protobuf_message)
        return img

    def protobufMessageToNumpy(self, img):
        print(str(img.width) + " x " + str(img.height))
        # print(img.pixels)
        sys.stdout.flush()

        npimg = np.array(img.pixels)
        npimg = npimg.reshape(1, img.height, img.width, 1)
        return (img.width, img.height, npimg)

    def dotsAndBoxesToCategorical(self, inputData):
        inp = np.copy(inputData)
        inp[inp == 255] = 1  # Line - comes first so that target data only has two categories
        inp[inp == 65] = 2  # Box A
        inp[inp == 150] = 3  # Box B
        inp[inp == 215] = 4  # Dot
        cat = to_categorical(inp)
        newShape = inp.shape + (cat.shape[-1],)
        return cat.reshape(newShape)

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

    def predict(self, message):
        print("received message")
        sys.stdout.flush()

        (imgWidth, imgHeight, npimgOrig) = self.protobufMessageToNumpy(message)
        npimg = self.dotsAndBoxesToCategorical(npimgOrig[:,:,:,0])

        print("request: ")
        print(np.floor(npimg[0, :, :, 4] * 100).astype(np.uint8))
        sys.stdout.flush()

        prediction_lines = self.model.predict(npimg)
        prediction = self.linesToDotsAndBoxesImage(prediction_lines[0], imgWidth, imgHeight)

        prediction_data_print = prediction * 100
        prediction_data_print = prediction_data_print.astype(np.uint8)
        print("prediction: ")
        print(prediction_data_print)
        sys.stdout.flush()

        prediction_data_img = prediction * 255
        prediction_data_img = prediction_data_img.astype(np.uint8)

        # merge image data in color channels
        tmp = np.zeros((imgHeight, imgWidth), dtype=np.uint8)
        merged_imgdata = np.stack([npimgOrig[0, :, :, 0].astype(np.uint8), prediction_data_img, tmp], axis=-1)

        # create image
        imgDebug = Image.fromarray(merged_imgdata, 'RGB')
        imgDebug = imgDebug.resize(size=(imgDebug.size[0] * 10, imgDebug.size[1] * 10))
        imgDebug.save('/tmp/debug.png')
        # imgDebug.show()

        # send data
        prediction_data_pb = pb.DotsAndBoxesImage()
        prediction_data_pb.width = imgWidth
        prediction_data_pb.height = imgHeight

        for y in range(imgHeight):
            for x in range(imgWidth):
                prediction_data_pb.pixels.append(prediction_data_img[y, x])

        return prediction_data_pb
