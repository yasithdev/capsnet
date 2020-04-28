import tensorflow as tf


@tf.function
def margin_loss(_y_true, _y_pred, _m_p=0.9, _m_n=0.1, _lambda=0.5):
    """
    Loss Function
    :param _y_true: shape: (None, num_caps)
    :param _y_pred: shape: (None, num_caps)
    :param _m_p: threshold for positive
    :param _m_n: threshold for negative
    :param _lambda: loss weight for negative
    :return: margin loss. shape: (None, )
    """
    p_err = tf.maximum(0., _m_p - _y_pred)  # shape: (None, num_caps)
    n_err = tf.maximum(0., _y_pred - _m_n)  # shape: (None, num_caps)
    p_loss = _y_true * tf.square(p_err)  # shape: (None, num_caps)
    n_loss = (1.0 - _y_true) * tf.square(n_err)  # shape: (None, num_caps)
    loss = tf.reduce_mean(p_loss + _lambda * n_loss, axis=-1)  # shape: (None, )
    return loss
