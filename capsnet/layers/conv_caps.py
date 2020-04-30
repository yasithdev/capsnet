import tensorflow as tf
from tensorflow import keras as k


class ConvCaps(k.layers.Layer):
    def __init__(self, filters, filter_dims, kernel_size, strides=(1, 1), padding='valid', activation=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.filter_dims = filter_dims
        self.conv_layer = k.layers.Conv2D(
            filters=self.filters * self.filter_dims,
            kernel_size=kernel_size,
            kernel_initializer=k.initializers.TruncatedNormal(),
            strides=strides,
            padding=padding,
            activation=activation)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'filter_dims': self.filter_dims
        })
        return config

    def call(self, inputs, **kwargs):
        result = tf.reshape(inputs, (-1, inputs.shape[1], inputs.shape[2], tf.reduce_prod(inputs.shape[3:])))
        result = self.conv_layer(result)
        result = tf.reshape(result, shape=(-1, *result.shape[1:3], self.filters, self.filter_dims))
        return result  # shape: (batch_size, rows, columns, filters, filter_dims)
