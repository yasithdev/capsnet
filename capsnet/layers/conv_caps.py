import tensorflow as tf
from tensorflow import keras as k

from capsnet.nn import squash


class ConvCaps(k.layers.Conv2D):
    def __init__(self, filters, filter_dims, kernel_size, **kwargs):
        self.filters = filters
        self.filter_dims = filter_dims
        super(ConvCaps, self).__init__(self.filters * self.filter_dims, kernel_size, **kwargs)

    def call(self, inputs, **kwargs):
        result = super(ConvCaps, self).call(inputs)
        result = tf.reshape(result, shape=(-1, *result.shape[1:3], result.shape[3] // self.filter_dims, self.filter_dims))
        activation = squash(result, axis=tf.constant((-1)))
        return activation
