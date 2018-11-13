# # AlphaZero version 10
# 
# This AlphaZero version starts from the weights of version 7 and is trained on augmented StageFour data.
# This version is called repeatedly by KSquares with fresh training data generated from the latest V10 network.

# In[1]:
import argparse
import sys
import time
sys.path.append('..')

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
import keras.backend as K
from keras.regularizers import l2
from keras.engine.topology import Layer

from PIL import Image
import random
import gc

from LineFilterLayer import LineFilterLayer
from ValueLayer import ValueLayer
from AugmentationSequence import AugmentationSequence

from upload import Upload

# parse command line args
parser = argparse.ArgumentParser(description='Train the alpha zero model on a given dataset')
parser.add_argument('--dataset')
parser.add_argument('--iteration', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--logdest', default='logs')
parser.add_argument('--initmodel', default='model/alphaZeroV7.h5')
parser.add_argument('--targetmodel', default='model/alphaZeroV10.h5')
parser.add_argument('--upload', action='store_true')
parser.add_argument('--no-augmentation', action='store_true')
args = parser.parse_args()

datasetPath = args.dataset
iteration = args.iteration
epochs = args.epochs
logdest = args.logdest
initial_model = args.initmodel
target_model = args.targetmodel
upload = args.upload
no_augmentation = args.no_augmentation


# read model from baseModelPath
baseModelPath = initial_model

# write result to target_model
modelPath = target_model

### make sure that basic assumptions are correct
assert(K.image_data_format() == 'channels_last')


def dotsAndBoxesToCategorical(inputData):
    inp = np.copy(inputData)
    inp[inp == 255] = 1 # Line - comes first so that target data only has two categories
    inp[inp == 65] = 2 # Box A
    inp[inp == 150] = 3 # Box B
    inp[inp == 215] = 4 # Dot
    cat = to_categorical(inp)
    newShape = inp.shape + (cat.shape[-1],)
    return cat.reshape(newShape)


def imgSizeToBoxes(x):
    return (x-3)/2


def lineFilterMatrixNP(imgWidth,imgHeight):
    boxWidth = imgSizeToBoxes(imgWidth)
    boxHeight = imgSizeToBoxes(imgHeight)
    linesCnt = 2*boxWidth*boxHeight+boxWidth+boxHeight
    mat = np.zeros((imgHeight, imgWidth), dtype=np.bool)
    for idx in range(linesCnt):
        y1 = idx / ((2*boxWidth) + 1)
        if idx % ((2*boxWidth) + 1) < boxWidth:
            # horizontal line
            x1 = idx % ((2*boxWidth) + 1)
            x2 = x1 + 1
            y2 = y1
        else:
            # vertical line
            x1 = idx % ((2*boxWidth) + 1) - boxWidth
            x2 = x1
            y2 = y1 + 1
        px = x2 * 2 + y2 - y1
        py = y2 * 2 + x2 - x1
        mat[py,px] = 1
    return mat


def loadRawPVDataset(datasetPath):
    try:
        rawDataset = np.load(datasetPath)
    except IOError:
        print("IOError: failed to load dataset!")
        exit(1)

    x_input = rawDataset['input']
    y_policy = rawDataset['policy']
    y_value = rawDataset['value']
    
    return (x_input, y_policy, y_value)


def process_input(x_input):
    return dotsAndBoxesToCategorical(x_input)


def process_policy(y_policy):
    y_policy = y_policy[:,lineFilterMatrixNP(y_policy.shape[-1], y_policy.shape[-2])]
    y_policy /= 255
    return y_policy


### setup logging
class Logger(object):
    def __init__(self, destfile):
        self.terminal = sys.stdout
        self.log = open(destfile, "a+")

    def write(self, message):
        #self.terminal.write(message)
        #self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.log.flush()
        self.terminal.flush()
        pass


timeStr = time.strftime("%Y.%m.%d-%H:%M")
logpath = logdest + "/" + os.path.basename(target_model).replace('.h5','') + "-" + timeStr + "." + str(iteration) + ".log"
logger = Logger(logpath)
sys.stderr = logger
sys.stdout = logger



print("training model configuration:")
print("|-> dataset path: " + str(datasetPath))
print("|-> iteration: " + str(iteration))
print("|-> epochs: " + str(epochs))
print("|-> logdest: " + str(logdest))
print("|-> initial model: " + str(initial_model))
print("|-> target model: " + str(target_model))
print("|-> upload: " + str(upload))
print("|-> augmentation: " + str(not no_augmentation))

print("logpath: " + logpath)
print("model path: " + modelPath)
print("base iteration model path: " + baseModelPath)

### load data
np.set_printoptions(precision=2)
(x_input_raw, y_policy_raw, y_value_raw) = loadRawPVDataset(datasetPath)

print("data shapes:")
print(x_input_raw.shape)
print(y_policy_raw.shape)
print(y_value_raw.shape)

print("example data [0]:")
print("raw input:")
print(x_input_raw[0])
print("raw policy:")
print(y_policy_raw[0])
print('raw value:')
print(y_value_raw[0])


### load the model
imgWidth = x_input_raw.shape[-1]
imgHeight = x_input_raw.shape[-2]
# LineFilterLayer has to be set before loading the model
LineFilterLayer.imgWidth = imgWidth
LineFilterLayer.imgHeight = imgHeight
# ValueLayer has to be set before loading the model
ValueLayer.imgWidth = imgWidth
ValueLayer.imgHeight = imgHeight

print("loading model: " + baseModelPath)
model = load_model(baseModelPath,
                   custom_objects={'LineFilterLayer':LineFilterLayer,
                   'ValueLayer':ValueLayer})

model.summary()


### Training
batch_size = 16
if no_augmentation:
    print("no augmentation")
    x_input = process_input(x_input_raw)
    y_policy = process_policy(y_policy_raw)
    y_value = y_value_raw
    model.fit(x_input, [y_policy, y_value], epochs=epochs, batch_size=batch_size)
else:
    print("augmentation")
    data_generator = AugmentationSequence(x_input_raw, y_policy_raw, y_value_raw, batch_size, process_input, process_policy)
    model.fit_generator(data_generator, epochs=epochs, steps_per_epoch=len(data_generator))

### Save model
model.save(modelPath)


def linesToDotsAndBoxesImage(lines, imgWidth, imgHeight):
    boxWidth = imgSizeToBoxes(imgWidth)
    boxHeight = imgSizeToBoxes(imgHeight)
    linesCnt = 2*boxWidth*boxHeight+boxWidth+boxHeight
    mat = np.zeros((imgHeight, imgWidth), dtype=lines.dtype)
    for idx in range(linesCnt):
        y1 = idx / ((2*boxWidth) + 1)
        if idx % ((2*boxWidth) + 1) < boxWidth:
            # horizontal line
            x1 = idx % ((2*boxWidth) + 1)
            x2 = x1 + 1
            y2 = y1
        else:
            # vertical line
            x1 = idx % ((2*boxWidth) + 1) - boxWidth
            x2 = x1
            y2 = y1 + 1
        px = x2 * 2 + y2 - y1
        py = y2 * 2 + x2 - x1
        mat[py,px] = lines[idx]
    return mat


### log an example image

example = random.randrange(x_input_raw.shape[0])
print("example: "+str(example))

input_data = process_input(x_input_raw[example:example+1])

(prediction_lines, prediction_value) = model.predict(input_data)
prediction_lines_print = prediction_lines * 100
print(prediction_lines_print.astype(np.uint8))
print(np.sum(prediction_lines))
prediction = linesToDotsAndBoxesImage(prediction_lines[0], imgWidth, imgHeight)

# print input data
input_data_print = x_input_raw[example]
print("input "+str(input_data_print.shape)+": ")
print(input_data_print)

# generate greyscale image data from input data
input_imgdata = x_input_raw[example]

# print prediction
prediction_data_print = prediction * 100 
prediction_data_print = prediction_data_print.astype(np.uint8)
print("prediction policy: ")
print(prediction_data_print)

print("prediction value: ")
print(prediction_value)

print("target value: ")
print(y_value_raw[example])

# generate greyscale image data from prediction data
prediction_imgdata = prediction * 255
prediction_imgdata = prediction_imgdata.astype(np.uint8)

# generate greyscale image of target data
target_imgdata = y_policy_raw[example]

# merge image data in color channels
merged_imgdata = np.stack([input_imgdata, prediction_imgdata, target_imgdata], axis=2)

#create image
img = Image.fromarray(merged_imgdata, 'RGB')
img = img.resize(size=(img.size[0]*10, img.size[1]*10))

imgpath = logdest + "/" + os.path.basename(target_model).replace('.h5','') + "." + str(iteration) + ".png"
img.save(imgpath)

# upload results
if upload:
    uploader = Upload()
    uploader.upload(modelPath, logpath, imgpath)
