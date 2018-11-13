import argparse
import sys

from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.regularizers import l2

from models.LineFilterLayer import LineFilterLayer
from models.ValueLayer import ValueLayer

from modelList import readYAMLModelList, Model as AlphaDotsModel, saveModelToYAML


# parse command line args
parser = argparse.ArgumentParser(description='Create a new AlphaZero model')
parser.add_argument('--name', required=True)
parser.add_argument('--desc', required=True)
parser.add_argument('--resblocks', type=int, required=True)
parser.add_argument('--filters', type=int, required=True)
parser.add_argument('--kernelsize', type=int, required=True)
parser.add_argument('--withmctsai', action='store_true')
args = parser.parse_args()


### setup logging
class Logger(object):
    def __init__(self, destfile):
        self.terminal = sys.stdout
        self.log = open(destfile, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.log.flush()
        self.terminal.flush()
        pass


timeStr = time.strftime("%Y.%m.%d-%H:%M")
logpath = "models/alphaZero/logs/createModel-" + args.name + "-" + timeStr + ".log"
sys.stderr = sys.stdout = Logger(logpath)

# image size doesn't really matter...
imgWidth = 11
imgHeight = 11

kernelSize = (args.kernelsize, args.kernelsize)
filterCnt = args.filters
l2reg = 1e-4
resBlockCnt = args.resblocks

print("createModel configuration:")
print("|-> model name: "+args.name)
print("|-> kernel size: "+str(args.kernelsize))
print("|-> filters: "+str(args.filters))
print("|-> res blocks: "+str(args.resblocks))
print("|-> with mcts ai: "+str(args.withmctsai))

for model in readYAMLModelList():
    if model['name'] == args.name:
        print("ERROR: name is already in use!")
        exit(1)



### build the model


def build_residual_block(x, index):
    in_x = x
    res_name = "res" + str(index)
    x = Conv2D(filters=filterCnt, kernel_size=kernelSize, padding="same",
               data_format="channels_last", kernel_regularizer=l2(l2reg),
               name=res_name + "_conv1_" + str(filterCnt))(x)
    x = BatchNormalization(name=res_name + "_batchnorm1")(x)
    x = Activation("relu", name=res_name + "_relu1")(x)
    x = Conv2D(filters=filterCnt, kernel_size=kernelSize, padding="same",
               data_format="channels_last", kernel_regularizer=l2(l2reg),
               name=res_name + "_conv2-" + str(filterCnt))(x)
    x = BatchNormalization(name="res" + str(index) + "_batchnorm2")(x)
    x = Add(name=res_name + "_add")([in_x, x])
    x = Activation("relu", name=res_name + "_relu2")(x)
    return x


img_input = Input(shape=(None, None, 5,))
x = Conv2D(filterCnt, kernelSize, padding='same', kernel_regularizer=l2(l2reg), name="input_conv")(img_input)
x = Activation("relu", name="input_relu")(x)
x = BatchNormalization()(x)

for i in range(resBlockCnt):
    x = build_residual_block(x, i + 1)

res_out = x

# policy output
x = Conv2D(1, kernelSize, padding='same', kernel_regularizer=l2(l2reg), name="policy_conv")(x)
x = LineFilterLayer(imgWidth, imgHeight)(x)
x = Activation("softmax", name="policy")(x)
policy_output = x

# value output
x = Conv2D(1, kernelSize, padding='same', kernel_regularizer=l2(l2reg), name="value_conv")(res_out)
# x = Flatten()(x)
# x = Dense(1, trainable=False, kernel_initializer=Constant(1.0/(imgWidth*imgHeight)), use_bias=False, name="value_dense")(x)
x = ValueLayer(imgWidth, imgHeight)(x)
x = Activation("tanh", name="value")(x)
value_output = x

model = Model(inputs=img_input, outputs=[policy_output, value_output])
model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])


model.summary()

### save the model and add it to yaml
modelPath = 'alphaZero/model/' + args.name + '.h5'
model.save("models/" + modelPath)


ad_model = AlphaDotsModel(args.name, args.desc, modelPath, 'PolicyValue', 'ConvNet')
saveModelToYAML(ad_model)
if args.withmctsai:
    ad_modelMCTS = AlphaDotsModel(args.name + "_MCTS", args.desc, modelPath, 'PolicyValue', 'MCTS-AlphaZero')
    saveModelToYAML(ad_modelMCTS)
