import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer

'''
This layer takes n_samples of Dots and Boxes images and returns a single value
input: (n_samples, img_height, img_width)
output: (n_samples, 1) 
'''
class ValueLayer(Layer):
    imgWidth = None
    imgHeight = None

    def __init__(self, imgWidth=None, imgHeight=None, **kwargs):
        if imgWidth is None or imgHeight is None:
            self.init(ValueLayer.imgWidth, ValueLayer.imgHeight)
        else:
            self.init(imgWidth, imgHeight)
        super(ValueLayer, self).__init__(**kwargs)

    def init(self, imgWidth, imgHeight):
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
        print("ValueLayer with image size {0} x {1}".format(imgWidth, imgHeight))

    def get_config(self):
        config = super(ValueLayer, self).get_config()
        config['img_width'] = self.imgWidth
        config['img_height'] = self.imgHeight
        return config

    @staticmethod
    def from_config(config, **kwargs):
        if ValueLayer.imgWidth is not None and ValueLayer.imgHeight is not None:
            config['img_width'] = ValueLayer.imgWidth
            config['img_height'] = ValueLayer.imgHeight
        return ValueLayer(config['img_width'], config['img_height'])

    def build(self, input_shape):
        shape = list(input_shape)
        assert (len(shape) == 4 and shape[3] == 1) or len(shape) == 3
        super(ValueLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        shape = K.shape(x)
        if shape[0] is None:
            return K.constant([None, 1])
        tf.assert_rank(x, 4)
        img_width = self.imgWidth
        img_height = self.imgHeight
        flat_x = K.reshape(x, (shape[0], img_width*img_height))
        val = 1.0 / (img_width * img_height)
        new_shape = [img_width * img_height, 1]
        kernel = K.constant(val, shape=new_shape)
        return K.dot(flat_x, kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
