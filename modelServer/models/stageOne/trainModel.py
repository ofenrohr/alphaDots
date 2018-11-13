
import numpy as np
import tensorflow as tf
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
import keras.backend as K
from PIL import Image
import random

modelPath = 'model/stageOne.h5'
modelCheckpointPath = 'model/train-checkpoint.h5'


np.set_printoptions(precision=2)

#
# Setup the model
#
img_input = Input(shape=(None,None,1,))
kernelSize = (5,5)
x = Conv2D(64, kernelSize, padding='same', activation='relu')(img_input)
x = BatchNormalization()(x)
x = Conv2D(64, kernelSize, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, kernelSize, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, kernelSize, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, kernelSize, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, kernelSize, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(2, kernelSize, padding='same', activation='softmax')(x)

model = Model(inputs=img_input, outputs=x)
model.compile(optimizer='sgd', loss='categorical_crossentropy', sample_weight_mode="temporal")

if os.path.isfile(os.path.join(os.path.curdir, modelPath)):
    print("loading model")
    model.load_weights(modelPath)
else:
    if os.path.isfile(os.path.join(os.path.curdir, modelCheckpointPath)):
        print("loading model checkpoint")
        model.load_weights(modelCheckpointPath)

model.summary()

#
# Load the data
#
firstTryDataset = np.load('stageOne5x4hard.npz')
x_train = firstTryDataset['x_train']
y_train = firstTryDataset['y_train']

print("original data:")
print(x_train[0])
print(y_train[0])
print(x_train.shape)
print(y_train.shape)


print("\nnormalized data:")
sp = x_train.shape
x_train = x_train.reshape((sp[0],sp[1],sp[2],1))
sp = y_train.shape
y_train = y_train.reshape((sp[0],sp[1],sp[2],1))

x_train = x_train.astype(K.floatx())
y_train = y_train.astype(K.floatx())
x_train /= 255
y_train /= 255

print(np.transpose(x_train[0]))
print(np.transpose(y_train[0]))
print(x_train.shape)
print(y_train.shape)

y_cat = to_categorical(y_train).reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 2)
print(y_cat.shape)
print(y_cat[0,:,:,0])
print(y_cat[0,:,:,1])


class_weights = np.zeros((y_train.shape[1], y_train.shape[2], 2))
class_weights[:, :, 0] += 0.1
class_weights[:, :, 1] += 0.9
#
# Train the model
#

callbacks = []

checkpoint = ModelCheckpoint(filepath=modelCheckpointPath, save_weights_only=False)
callbacks.append(checkpoint)

progbar = ProgbarLogger()
callbacks.append(progbar)

tensorboard = TensorBoard(log_dir='model/log', write_grads=True, write_graph=True, write_images=True, histogram_freq=1)
callbacks.append(tensorboard)

model.fit(x_train, y_cat, epochs=50, batch_size=64, callbacks=callbacks, validation_split=0.001, sample_weight=y_cat[:,:,:,1].reshape(y_cat.shape[0], y_cat.shape[1] * y_cat.shape[2]))

model.save(modelPath)