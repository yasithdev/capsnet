import tensorflow as tf
from tensorflow.keras import backend as kb


@tf.function
def softmax(_logits, axis):
    return tf.exp(_logits) / tf.reduce_sum(tf.exp(_logits), axis, keepdims=True)


@tf.function
def norm(data, axis=-1):
    e = kb.epsilon()
    squared_norm = kb.sum(kb.square(data), axis=axis, keepdims=False)
    return kb.sqrt(squared_norm)


def squash(data, axis):
    """
    Normalize to unit vectors
    :param e: small constant for numerical stability
    :param data: Tensor with rank >= 2
    :param axis: axis over which to squash
    :return:
    """
    # e = kb.epsilon()
    squared_norm = kb.sum(kb.square(data), axis=axis, keepdims=True)
    scale = squared_norm / (1 + squared_norm)
    unit = data / kb.sqrt(squared_norm)
    return scale * unit


def mask(inputs):
    """
    Mask data from all capsules except the most activated one, for each instance
    :param inputs: shape: (None, num_caps, dim_caps)
    :return:
    """
    norm_ = norm(inputs, axis=-1)  # shape: (None, num_caps)
    argmax = tf.argmax(norm_, axis=-1)  # shape: (None, )
    mask_ = tf.expand_dims(tf.one_hot(argmax, depth=norm_.shape[-1]), axis=-1)  # shape: (None, num_caps, 1)
    masked_input = tf.multiply(inputs, mask_)  # shape: (None, num_caps, dim_caps)
    return masked_input


def mask_cid(inputs):
    """
    Select most activated capsule from each instance and return it
    :param inputs: shape: (None, num_caps, dim_caps)
    :return:
    """
    norm_ = norm(inputs, axis=-1)  # shape: (None, num_caps)
    # build index of elements to collect
    i = tf.range(start=0, limit=tf.shape(inputs)[0], delta=1)  # shape: (None, )
    j = tf.argmax(norm_, axis=-1)  # shape: (None, )
    ij = tf.stack([i, tf.cast(j, tf.int32)], axis=1)
    # gather from index and return
    return tf.gather_nd(inputs, ij)
