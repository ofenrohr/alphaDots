import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer

'''
This layer takes n_samples of Dots and Boxes images and returns
only the line pixels in a flattened list
input: (n_samples, img_height, img_width)
output: (n_samples, 2*((w-3)/2)*((h-3)/2) + ((w-3)/2) + ((h-3)/2)) 
where w is img_width and h is img_height
'''
class LineFilterLayer(Layer):
    imgWidth = None
    imgHeight = None
    noShapeCheck = None

    def __init__(self, imgWidth=None, imgHeight=None, noShapeCheck=False, **kwargs):
        if imgWidth is None or imgHeight is None:
            #self.filterMatrix = None
            #self.output_dim = None
            self.init(LineFilterLayer.imgWidth, LineFilterLayer.imgHeight, noShapeCheck)
        else:
            self.init(imgWidth, imgHeight, noShapeCheck)
        super(LineFilterLayer, self).__init__(**kwargs)

    def init(self, imgWidth, imgHeight, noShapeCheck):
        print("LineFilterLayer with image size %d x %d" % (imgWidth, imgHeight))
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
        self.filterMatrix = LineFilterLayer.lineFilterMatrix(imgWidth, imgHeight)
        w = imgWidth
        h = imgHeight
        self.output_dim = tf.constant(2 * ((w - 3) / 2) * ((h - 3) / 2) + ((w - 3) / 2) + ((h - 3) / 2))
        self.noShapeCheck = noShapeCheck

    def get_config(self):
        config = super(LineFilterLayer, self).get_config()
        config['img_width'] = self.imgWidth
        config['img_height'] = self.imgHeight
        config['no_shape_check'] = self.noShapeCheck
        return config

    @staticmethod
    def from_config(config, **kwargs):
        if LineFilterLayer.imgWidth is not None and LineFilterLayer.imgHeight is not None:
            config['img_width'] = LineFilterLayer.imgWidth
            config['img_height'] = LineFilterLayer.imgHeight
        if LineFilterLayer.noShapeCheck is not None:
            config['no_shape_check'] = LineFilterLayer.noShapeCheck
        if 'no_shape_check' not in config:
            config['no_shape_check'] = False
        return LineFilterLayer(config['img_width'], config['img_height'], config['no_shape_check'])

    def build(self, input_shape):
        shape = list(input_shape)
        if not self.noShapeCheck:
            assert len(shape) == 4
            assert shape[3] == 1
        self.compute_output_shape(input_shape)
        super(LineFilterLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        shapeX = K.shape(x)
        if shapeX[0] is None:
            return tf.constant([None, None])
        tf.assert_rank(x, 4)
        assert self.filterMatrix is not None
        assert self.output_dim is not None
        #assert self.output_dim > 0
        #tf.assert_rank_at_least(self.output_dim, 1)
        lines2D = tf.boolean_mask(x, self.filterMatrix, name='lineFilter', axis=1)
        shape = K.shape(lines2D)
        # print(shape)
        # print(shape[0])
        #return K.reshape(lines2D, (shape[0], self.output_dim))
        return K.reshape(lines2D, shape[0:2])
        #return lines2D

    def compute_output_shape(self, input_shape):
        w = input_shape[-2]
        h = input_shape[-3]
        if w is None or h is None:
            return (input_shape[0], None)
        self.filterMatrix = LineFilterLayer.lineFilterMatrix(w, h)
        wbox = LineFilterLayer.imgSizeToBoxes(w)
        hbox = LineFilterLayer.imgSizeToBoxes(h)
        self.output_dim = 2 * wbox * hbox + wbox + hbox
        return (input_shape[0], self.output_dim)

    @staticmethod
    def imgSizeToBoxes(x):
        if x is None:
            return None
        return int((x - 3) / 2)

    @staticmethod
    def lineFilterMatrix(imgWidth, imgHeight, asnumpy=False):
        boxWidth = LineFilterLayer.imgSizeToBoxes(imgWidth)
        boxHeight = LineFilterLayer.imgSizeToBoxes(imgHeight)
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
            px = int(x2 * 2 + y2 - y1)
            py = int(y2 * 2 + x2 - x1)
            mat[py, px] = 1

        if asnumpy:
            return mat

        return tf.convert_to_tensor(mat)
