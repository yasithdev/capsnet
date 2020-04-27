import tensorflow as tf
from tensorflow import keras as k


@tf.function
def softmax(_logits, axis):
    return tf.exp(_logits) / tf.reduce_sum(tf.exp(_logits), axis, keepdims=True)


@tf.function
def norm(data, axis=-1):
    e = k.backend.epsilon()
    squared_norm = tf.reduce_sum(tf.square(data), axis=axis, keepdims=False)
    return tf.sqrt(squared_norm + e)


def squash(data, axis):
    """
    Normalize to unit vectors
    :param data: Tensor with rank >= 2
    :param axis: axis over which to squash
    :return:
    """
    e = k.backend.epsilon()
    squared_norm = tf.reduce_sum(tf.square(data), axis=axis, keepdims=True)
    return (squared_norm / (1 + squared_norm)) * (data / tf.sqrt(squared_norm + e))
