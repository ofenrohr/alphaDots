import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer

'''
This layer takes n_samples of Dots and Boxes images and returns
only the line pixels in a flattened list
input: (n_samples, n_frames, img_height, img_width)
output: (n_samples, n_frames, 2*((w-3)/2)*((h-3)/2) + ((w-3)/2) + ((h-3)/2)) 
where w is img_width and h is img_height
'''
class SequenceLineFilterLayer(Layer):
    imgWidth = None
    imgHeight = None
    noShapeCheck = None

    def __init__(self, imgWidth=None, imgHeight=None, noShapeCheck=False, **kwargs):
        if imgWidth is None or imgHeight is None:
            #self.filterMatrix = None
            #self.output_dim = None
            print("using static image size %d x %d" % SequenceLineFilterLayer.imgWidth, SequenceLineFilterLayer.imgHeight)
            self.init(SequenceLineFilterLayer.imgWidth, SequenceLineFilterLayer.imgHeight, noShapeCheck)
        else:
            self.init(imgWidth, imgHeight, noShapeCheck)
        super(SequenceLineFilterLayer, self).__init__(**kwargs)

    def init(self, imgWidth, imgHeight, noShapeCheck):
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
        self.filterMatrix = SequenceLineFilterLayer.lineFilterMatrix(imgWidth, imgHeight)
        w = imgWidth
        h = imgHeight
        self.output_dim = 2 * ((w - 3) / 2) * ((h - 3) / 2) + ((w - 3) / 2) + ((h - 3) / 2)
        self.noShapeCheck = noShapeCheck

    def get_config(self):
        config = super(SequenceLineFilterLayer, self).get_config()
        config['img_width'] = self.imgWidth
        config['img_height'] = self.imgHeight
        config['no_shape_check'] = self.noShapeCheck
        return config

    @staticmethod
    def from_config(config, **kwargs):
        print("SequenceLineFilterLayer from_config!")
        if SequenceLineFilterLayer.imgWidth is not None and SequenceLineFilterLayer.imgHeight is not None:
            config['img_width'] = SequenceLineFilterLayer.imgWidth
            config['img_height'] = SequenceLineFilterLayer.imgHeight
        if SequenceLineFilterLayer.noShapeCheck is not None:
            config['no_shape_check'] = SequenceLineFilterLayer.noShapeCheck
        print("using static image size {0} x {1}".format(SequenceLineFilterLayer.imgWidth, SequenceLineFilterLayer.imgHeight))
        return SequenceLineFilterLayer(config['img_width'], config['img_height'], config['no_shape_check'])

    def build(self, input_shape):
        shape = list(input_shape)
        if not self.noShapeCheck:
            assert len(shape) == 5
            assert shape[-1] == 1
        self.compute_output_shape(input_shape)
        super(SequenceLineFilterLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        shapeX = K.shape(x)
        if shapeX[0] is None:
            return tf.constant([None] * (tf.size(shapeX)-2))
        assert self.filterMatrix is not None
        assert self.output_dim is not None
        assert self.output_dim > 0
        #assert shapeX[4] == 1 or shapeX[4] is None
        lines2D = tf.boolean_mask(tf.reshape(x, (shapeX[0], shapeX[1], shapeX[2], shapeX[3]), name='lfReshape1'), self.filterMatrix, name='lineFilter', axis=2)
        shape = K.shape(lines2D)
        # print(shape)
        # print(shape[0])
        return tf.reshape(lines2D, (shapeX[0], shapeX[1], self.output_dim), name='lfReshape2')

    def compute_output_shape(self, input_shape):
        w = input_shape[-2]
        h = input_shape[-3]
        if w is None or h is None:
            return (input_shape[0], None, None)
        self.filterMatrix = SequenceLineFilterLayer.lineFilterMatrix(w, h)
        wbox = SequenceLineFilterLayer.imgSizeToBoxes(w)
        hbox = SequenceLineFilterLayer.imgSizeToBoxes(h)
        self.output_dim = 2 * wbox * hbox + wbox + hbox
        return (input_shape[0], input_shape[1], self.output_dim)

    @staticmethod
    def imgSizeToBoxes(x):
        if x is None:
            return None
        return (x - 3) / 2

    @staticmethod
    def lineFilterMatrix(imgWidth, imgHeight, asnumpy=False):
        boxWidth = SequenceLineFilterLayer.imgSizeToBoxes(imgWidth)
        boxHeight = SequenceLineFilterLayer.imgSizeToBoxes(imgHeight)
        linesCnt = 2 * boxWidth * boxHeight + boxWidth + boxHeight
        mat = np.zeros((imgHeight, imgWidth), dtype=np.bool)
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
            mat[py, px] = 1

        if asnumpy:
            return mat

        return tf.convert_to_tensor(mat)
