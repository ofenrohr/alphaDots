
# coding: utf-8

# # AlphaZero version 5
# 
# This model was trained from scratch on various StageTwo Datasets for two epochs each.

# In[1]:

import sys


sys.path.append('..')

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
import keras.backend as K
from keras.regularizers import l2
from keras.engine.topology import Layer

from PIL import Image
from matplotlib.pyplot import imshow
#get_ipython().magic(u'matplotlib inline')
import random
import gc

from LineFilterLayer import LineFilterLayer
import DebugLogger

modelPath = 'model/alphaZeroV5.3.h5'

datasetList = [
    'StageTwo-1000-5x4-15:53-09_04_2018.npz',
    'StageTwo-1000000-6x5-23:21-08_04_2018.npz',
    'StageTwo-1000000-6x5-23:41-08_04_2018.npz',
    'StageTwo-1000000-6x5-08:32-09_04_2018.npz',
    'StageTwo-1000000-6x5-08:51-09_04_2018.npz',
    'StageTwo-1000000-6x5-09:10-09_04_2018.npz',
    'StageTwo-1000000-6x5-09:28-09_04_2018.npz',
    'StageTwo-1000000-6x5-09:47-09_04_2018.npz',
    'StageTwo-1000000-6x5-10:06-09_04_2018.npz',
    'StageTwo-1000000-6x5-10:25-09_04_2018.npz',
    'StageTwo-1000000-6x5-10:44-09_04_2018.npz',
    'StageTwo-1000000-6x5-11:03-09_04_2018.npz',
    'StageTwo-1000000-6x5-11:21-09_04_2018.npz',
]


# In[2]:

print(K.image_data_format()) 
# expected output: channels_last
assert(K.image_data_format() == 'channels_last' )


# In[5]:

def dotsAndBoxesToCategorical(inp):
    #inp = np.copy(inputData)
    inp[inp == 255] = 1 # Line - comes first so that target data only has two categories
    inp[inp == 65] = 2 # Box A
    inp[inp == 150] = 3 # Box B
    inp[inp == 215] = 4 # Dot
    oldShape = inp.shape
    inp = to_categorical(inp)
    newShape = oldShape + (inp.shape[-1],)
    return inp.reshape(newShape)


# In[6]:

def imgSizeToBoxes(x):
    return int((x-3)/2)

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
        px = int(x2 * 2 + y2 - y1)
        py = int(y2 * 2 + x2 - x1)
        mat[py,px] = 1
    return mat

#lineFilterMatrixNP(13,11)


# In[7]:

def loadDataset(datasetPath):
    rawDataset = np.load(datasetPath)
    
    x_train = rawDataset['x_train']
    y_train = rawDataset['y_train']
    
    x_train = dotsAndBoxesToCategorical(x_train)
    y_train = y_train[:,lineFilterMatrixNP(y_train.shape[-1], y_train.shape[-2])]
    y_train = np.divide(y_train, 255)
    
    return (x_train, y_train)

np.set_printoptions(precision=2)


# In[8]:

(x_train, y_train) = loadDataset(datasetList[0])

print(x_train.shape)
print(y_train.shape)


# In[9]:

kernelSize = (5,5)
filterCnt = 64
l2reg = 1e-4
resBlockCnt = 4
inputWidth = int(x_train.shape[-2])
inputHeight = int(x_train.shape[-3])

def build_residual_block(x, index):
        in_x = x
        res_name = "res"+str(index)
        x = Conv2D(filters=filterCnt, kernel_size=kernelSize, padding="same",
                   data_format="channels_last", kernel_regularizer=l2(l2reg), 
                   name=res_name+"_conv1_"+str(filterCnt))(x)
        x = BatchNormalization(name=res_name+"_batchnorm1")(x)
        x = Activation("relu",name=res_name+"_relu1")(x)
        x = Conv2D(filters=filterCnt, kernel_size=kernelSize, padding="same",
                   data_format="channels_last", kernel_regularizer=l2(l2reg), 
                   name=res_name+"_conv2-"+str(filterCnt))(x)
        x = BatchNormalization(name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=res_name+"_add")([in_x, x])
        x = Activation("relu", name=res_name+"_relu2")(x)
        return x


img_input = Input(shape=(None,None,5,))
x = Conv2D(filterCnt, kernelSize, padding='same', kernel_regularizer=l2(l2reg), name="input_conv")(img_input)
x = Activation("relu", name="input_relu")(x)
x = BatchNormalization()(x)

for i in range(resBlockCnt):
    x = build_residual_block(x, i+1)

res_out = x

x = Conv2D(1, kernelSize, padding='same', kernel_regularizer=l2(l2reg), name="output_conv")(x)
x = LineFilterLayer(inputWidth, inputHeight)(x)
x = Activation("softmax", name="output_softmax")(x)
    
model = Model(inputs=img_input, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.summary()
model.save(modelPath)


for layer in model.layers:
    print("{:30} {:50} {!s}".format(layer.name, str(layer.input_shape), str(layer.output_shape)))

# In[10]:

#sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#K.set_session(sess)

iteration = 1
for datasetPath in datasetList:
    print("cleaning up dataset")
    del x_train
    del y_train
    gc.collect()
    
    print("loading dataset " + datasetPath)
    (x_train, y_train) = loadDataset(datasetPath)
    
    print(x_train.shape)
    print(y_train.shape)
    
    # update the line filter layer to reflect new board size in dataset
    LineFilterLayer.imgWidth = x_train.shape[-2]
    LineFilterLayer.imgHeight = x_train.shape[-3]
    model = load_model(modelPath, custom_objects={'LineFilterLayer':LineFilterLayer}) 

    # Training
    callbacks = []

    checkpoint = ModelCheckpoint(filepath=modelPath+".checkpoint."+str(iteration), save_weights_only=False)
    callbacks.append(checkpoint)

    #progbar = ProgbarLogger()
    #callbacks.append(progbar)

    #tensorboard = TensorBoard(log_dir='model/log2', write_grads=True, write_graph=True, write_images=True, histogram_freq=1)
    #callbacks.append(tensorboard)

    #callbacks.append(DebugLogger.DebugLogger())

    model.fit(x_train, y_train, epochs=1, batch_size=64, callbacks=callbacks, validation_split=0.001)

    model.save(modelPath)
    
    iteration += 1


# In[10]:

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


# In[11]:

example = random.randrange(x_train.shape[0])
print("example: "+str(example))

input_data_cat = x_train[example:example+1]

prediction_lines = model.predict(input_data_cat)
prediction_lines_print = prediction_lines * 100
print(prediction_lines_print.astype(np.uint8))
print(np.sum(prediction_lines))
prediction = linesToDotsAndBoxesImage(prediction_lines[0], x_train.shape[2], x_train.shape[1])

# print input data
input_data_print = x_train[example,:,:] 
input_data_print = input_data_print.astype(np.uint8)
print("input "+str(input_data_print.shape)+": ")
print(input_data_print)

# generate greyscale image data from input data
target_imgdata = x_train[example,:,:] 
target_imgdata = target_imgdata.astype(np.uint8)

# print prediction
prediction_data_print = prediction * 100 
prediction_data_print = prediction_data_print.astype(np.uint8)
print("prediction: ")
print(prediction_data_print)

# generate greyscale image data from prediction data
prediction_imgdata = prediction * 255
prediction_imgdata = prediction_imgdata.astype(np.uint8)

# merge image data in color channels
tmp = np.zeros((prediction.shape[0], prediction.shape[1]), dtype=np.uint8)
merged_imgdata = np.stack([target_imgdata, prediction_imgdata, tmp], axis=2)

#create image
img = Image.fromarray(merged_imgdata, 'RGB')
img = img.resize(size=(img.size[0]*10, img.size[1]*10))

img.save("/tmp/example.png")

# In[ ]:



