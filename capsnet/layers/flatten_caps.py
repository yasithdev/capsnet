import tensorflow as tf
import tensorflow.keras as k


class FlattenCaps(k.layers.Layer):
  def __init__(self, caps, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    super(FlattenCaps, self).__init__(trainable, name, dtype, dynamic, **kwargs)
    self.caps = caps
    self.input_caps = ...
    self.input_caps_dims = ...
    self.w = ...

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'caps': self.caps,
    })
    return config

  def build(self, input_shape):
    # sanity check
    tf.assert_equal(input_shape.rank, 5, message=f'Expected Tensor of Rank = 5, Got Rank = {input_shape.rank}')
    # define capsule parameters
    self.input_caps = input_shape[1] * input_shape[2] * input_shape[3]
    self.input_caps_dims = input_shape[4]
    # define weights
    self.w = self.add_weight(
      name='w',
      shape=(1, self.caps, self.input_caps, 1),  # (1, c, c_in, 1)
      dtype=tf.float32,
      initializer=k.initializers.TruncatedNormal(),
    )
    self.built = True

  def call(self, inputs, **kwargs):
    inputs = tf.reshape(inputs, (-1, 1, self.input_caps, self.input_caps_dims))  # (b, 1, c_in, d)
    output = tf.reduce_sum(inputs * self.w, axis=-2)  # (b, c, c_in, d) -> (b, c, d)
    return output  # (b, c, d)
