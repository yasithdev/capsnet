import tensorflow as tf
import tensorflow.keras as k


class ConvCaps2D(k.layers.Layer):
  def __init__(self, filters, filter_dims, kernel_size, strides=(1, 1), padding='valid', **kwargs):
    super(ConvCaps2D, self).__init__(**kwargs)
    self.filters = filters
    self.filter_dims = filter_dims
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.conv_layer = ...  # initialize at build()

  def build(self, input_shape):
    self.conv_layer = k.layers.Conv2D(
      filters=self.filters * self.filter_dims,
      kernel_size=self.kernel_size,
      strides=self.strides,
      activation='linear',
      groups=input_shape[-1] // self.filter_dims,  # capsule-wise isolated convolution
      padding=self.padding
    )
    self.built = True

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'filters': self.filters,
      'filter_dims': self.filter_dims,
      'kernel_size': self.kernel_size,
      'strides': self.strides,
      'padding': self.padding
    })
    return config

  @tf.function(experimental_compile=True)
  def call(self, inputs, *args, **kwargs):
    result = tf.reshape(inputs, (-1, inputs.shape[1], inputs.shape[2], tf.reduce_prod(inputs.shape[3:])))
    result = self.conv_layer(result)
    result = tf.reshape(result, shape=(-1, *result.shape[1:3], self.filters, self.filter_dims))
    return result  # shape: (batch_size, rows, columns, filters, filter_dims)
