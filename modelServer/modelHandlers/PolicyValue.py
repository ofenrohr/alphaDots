import sys
sys.path.append('..')
from protobuf.py.DotsAndBoxesImage_pb2 import DotsAndBoxesImage
from protobuf.py.PolicyValueData_pb2 import PolicyValueData
import numpy as np
from keras.utils.np_utils import to_categorical
from PIL import Image
from Logger import Logger

class PolicyValue:
    def __init__(self, model, invertValue, debug=False, logdest='/tmp', logger=None):
        print("PolicyValue, debug: "+str(debug))
        self.model = model
        self.invertValue = invertValue
        self.batchIdx = 0
        self.batch = None
        self.imgWidth = None
        self.imgHeight = None
        self.debug = debug
        self.logDest = logdest
        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger(self.debug, self.logDest)
        self.categorical = False

    def setCategorical(self, categorical):
        self.categorical = categorical

    def deserializeProtobuf(self, protobuf_message):
        img = DotsAndBoxesImage()
        img.ParseFromString(protobuf_message)
        return img

    def protobufMessageToNumpySimple(self, img):
        return (img.width, img.height, np.array(img.pixels).reshape(1, img.height, img.width))

    def protobufMessageToNumpy(self, img):
        if self.debug:
            self.logger.write(str(img.width) + " x " + str(img.height))
            sys.stdout.flush()

        hasNextImg = True
        batchSize = 1
        npimgs = []
        while hasNextImg:
            npimg = np.array(img.pixels)
            npimgs.extend(npimg)

            if img.HasField("nextImage"):
                img = img.nextImage
                hasNextImg = True
                batchSize += 1
            else:
                hasNextImg = False

        retnpimg = np.array(npimgs).reshape(batchSize, img.height, img.width)
        return (img.width, img.height, retnpimg)

    def dotsAndBoxesToCategorical(self, inputData):
        if not self.categorical:
            inp = np.copy(inputData)
            inp[inp == 255] = 1  # Line - comes first so that target data only has two categories
            inp[inp == 65] = 2   # Box A
            inp[inp == 150] = 3  # Box B
            inp[inp == 215] = 4  # Dot
            cat = to_categorical(inp)
            newShape = inp.shape + (cat.shape[-1],)
            return cat.reshape(newShape)
        else:
            return to_categorical(inputData)

    def imgSizeToBoxes(self, x):
        if x is None:
            return None
        return (x - 3) / 2

    def linesToDotsAndBoxesImage(self, lines, imgWidth, imgHeight):
        boxWidth = self.imgSizeToBoxes(imgWidth)
        boxHeight = self.imgSizeToBoxes(imgHeight)
        linesCnt = int(2 * boxWidth * boxHeight + boxWidth + boxHeight)
        mat = np.zeros((lines.shape[0], imgHeight, imgWidth), dtype=lines.dtype)
        for sample in range(lines.shape[0]):
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
                mat[sample, py, px] = lines[sample, idx]
        return mat


    def predict(self, message):
        if self.debug:
            self.logger.write("received request, processing...")
            sys.stdout.flush()

        (imgWidth, imgHeight, npimgOrig) = self.protobufMessageToNumpySimple(message)

        if self.debug:
            self.logger.write("request shape: " + str(npimgOrig.shape))
            sys.stdout.flush()

        npimg = self.dotsAndBoxesToCategorical(npimgOrig)

        if self.debug:
            self.logger.write("categorical shape: " + str(npimg.shape))
            self.logger.write("request: ")
            self.logger.write(np.floor(npimgOrig).astype(np.uint8))
            sys.stdout.flush()

        # make prediction
        (prediction_policy, prediction_value) = self.model.predict(npimg)

        if self.debug:
            prediction_data_img = self.linesToDotsAndBoxesImage(prediction_policy, imgWidth, imgHeight) * 255
            prediction_data_img = prediction_data_img.astype(np.uint8)
            self.logger.write("prediction: ")
            self.logger.write(prediction_value)
            self.logger.write(prediction_data_img)
            sys.stdout.flush()

            for sample in range(prediction_policy.shape[0]):
                # merge image data in color channels
                tmp = np.zeros((imgHeight, imgWidth), dtype=np.uint8)
                merged_imgdata = np.stack([npimgOrig[sample].astype(np.uint8), prediction_data_img[sample], tmp], axis=-1)

                # create image
                imgDebug = Image.fromarray(merged_imgdata, 'RGB')
                imgDebug = imgDebug.resize(size=(imgDebug.size[0] * 10, imgDebug.size[1] * 10))
                imgDebug.save(self.logDest + '/PolicyValue-'+str(sample)+'.png')
                # imgDebug.show()

        # send data
        prediction_data_pb = PolicyValueData()
        # single prediction way:
        if self.invertValue:
            prediction_data_pb.value = -prediction_value[0]
        else:
            prediction_data_pb.value = prediction_value[0]

        for l in prediction_policy[0]:
            prediction_data_pb.policy.append(l)

        # original, batch way:
        '''
        tmp_pvd = prediction_data_pb
        for i in range(prediction_policy.shape[0]):
            if self.invertValue:
                tmp_pvd.value = -prediction_value[i]
            else:
                tmp_pvd.value = prediction_value[i]

            for l in prediction_policy[i]:
                tmp_pvd.policy.append(l)

            if i < prediction_policy.shape[0]-1:
                tmp_pvd.nextData.value = 0
                tmp_pvd = tmp_pvd.nextData
        '''

        return prediction_data_pb