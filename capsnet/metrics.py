import tensorflow as tf


@tf.function
def accuracy(_y_true, _y_pred):
    """
    :param _y_true: shape: (None, num_caps)
    :param _y_pred: shape: (None, num_caps)
    :return:
    """
    _y_pred = tf.argmax(_y_pred, axis=-1)
    _y_true = tf.argmax(_y_true, axis=-1)
    correct = tf.equal(_y_true, _y_pred)
    return tf.reduce_mean(tf.cast(correct, tf.float32))
