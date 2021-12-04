import tensorflow as tf
import tensorflow.keras as k


def softmax(_logits, axis):
  return tf.exp(_logits) / tf.reduce_sum(tf.exp(_logits), axis, keepdims=True)


@tf.function
def norm(data):
  e = 1e-10
  squared_sum = tf.reduce_sum(tf.square(data), axis=-1)
  return tf.sqrt(squared_sum + e)


def squash(data, axis=-1):
  """
  Normalize to unit vectors
  :param data: Tensor with rank >= 2
  :param axis: axis over which to squash
  :return:
  """
  e = 1e-10
  squared_sum = tf.reduce_sum(tf.square(data), axis=axis, keepdims=True)
  vec_norm = tf.sqrt(squared_sum + e)
  return squared_sum / (1 + squared_sum) * data / vec_norm


class Mask(k.layers.Layer):

  def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    super(Mask, self).__init__(trainable, name, dtype, dynamic, **kwargs)

  def call(self, inputs, *args, **kwargs):  # input_shape: (None, num_caps, dim_caps)
    # calculate capsule norms
    norms = norm(inputs)  # shape: (None, num_caps)
    # find capsule indices with largest norms
    indices = tf.argmax(norms, axis=-1, output_type=tf.int32)  # shape: (None, )
    # create a mask to apply to input
    mask = tf.expand_dims(tf.one_hot(indices, depth=norms.shape[-1]), axis=-1)  # shape: (None, num_caps, 1)
    # apply mask to input
    return tf.multiply(inputs, mask)  # shape: (None, num_caps, dim_caps)


class MaskCID(k.layers.Layer):

  def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    super(MaskCID, self).__init__(trainable, name, dtype, dynamic, **kwargs)

  def call(self, inputs, *args, **kwargs):  # input_shape: (None, num_caps, dim_caps)
    # calculate capsule norms
    norms = norm(inputs)  # shape: (None, num_caps)
    # find capsule indices with largest norms
    indices = tf.argmax(norms, axis=-1, output_type=tf.int32)  # shape: (None, )
    # gather largest capsules from input
    return tf.gather(inputs, indices, axis=1, batch_dims=1)  # shape: (None, dim_caps)
