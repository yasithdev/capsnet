import tensorflow as tf
from tensorflow import keras as k

from capsnet.nn import squash, softmax


@tf.function
def routing_step(_logits, _pre_activation):
    """
    Weight the prediction by routing weights, squash it, and return it
    :param _logits: (batch_size, p_num_caps, num_caps, 1, 1)
    :param _pre_activation: (batch_size, p_num_caps, num_caps, 1, dim_caps)
    :return:
    """
    # softmax of logits over all capsules (such that their sum is 1)
    _prob = softmax(_logits, axis=2)  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
    # calculate activation based on _prob
    _activation = tf.reduce_sum(_prob * _pre_activation, axis=1, keepdims=True)  # shape: (batch_size, 1, num_caps, 1, dim_caps)
    # squash over dim_caps and return
    return squash(_activation, axis=-1)  # shape: (batch_size, 1, num_caps, 1, dim_caps)


@tf.function
def routing_loop(_i, _logits, _pre_activation):
    # step 1: find the activation from logits
    _activation = routing_step(_logits, _pre_activation)  # shape: (batch_size, 1, num_caps, 1, dim_caps)
    # step 2: find the agreement (dot product) between pre_activation and activation, across dim_caps
    _agreement = tf.reduce_sum(_pre_activation * _activation, axis=-2, keepdims=True)  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
    # step 3: update routing weights based on agreement
    _logits = _logits + _agreement
    # return updated variables
    return _i + 1, _logits, _pre_activation


class DenseCaps(k.layers.Layer):
    def __init__(self, caps, caps_dims, routing_iter, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.caps = caps
        self.caps_dims = caps_dims
        self.routing_iter = routing_iter
        self.input_caps = ...
        self.input_caps_dims = ...
        self.w = ...

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'caps': self.caps,
            'caps_dims': self.caps_dims,
            'routing_iter': self.routing_iter,
            'input_caps': self.input_caps,
            'input_caps_dims': self.input_caps_dims
        })
        return config

    def build(self, input_shape: tf.TensorShape):
        # sanity check
        tf.assert_equal(input_shape.rank, 5, message=f'Expected Tensor of Rank = 5, Got Rank = {input_shape.rank}')
        # define capsule parameters
        self.input_caps = input_shape[1] * input_shape[2] * input_shape[3]
        self.input_caps_dims = input_shape[4]
        # define weights
        self.w = self.add_weight(
            name='w',
            shape=(1, self.input_caps, self.caps, self.input_caps_dims, self.caps_dims),
            dtype=tf.float32,
            initializer='random_normal'
        )
        self.built = True

    def call(self, inputs, **kwargs):
        # get batch size of input
        batch_size = tf.shape(inputs)[0]
        # reshape input
        inputs = tf.reshape(inputs, (batch_size, self.input_caps, 1, self.input_caps_dims, 1))  # shape: (batch_size, p_num_caps, 1, p_dim_caps, 1)
        # calculate pre-activation (dot product of the last two dimensions of tiled_w and input)
        pre_activation = tf.reduce_sum(inputs * self.w, axis=-2, keepdims=True)  # shape: (batch_size, p_num_caps, num_caps, 1, dim_caps)
        # dynamic routing
        activation = self.dynamic_routing(pre_activation=pre_activation)  # shape: (batch_size, 1, num_caps, 1, dim_caps)
        # reshape to (None, num_caps, dim_caps) and return
        return tf.reshape(activation, shape=(-1, self.caps, self.caps_dims))

    def dynamic_routing(self, pre_activation):
        """
        Dynamic Routing as proposed in the original paper

        :param pre_activation: shape: (batch_size, p_num_caps, num_caps, 1, dim_caps)
        :return:
        """
        tensor_shape = tf.shape(pre_activation)
        batch_size = tensor_shape[0]
        input_caps = tensor_shape[1]
        caps = tensor_shape[2]
        # define variables
        logits = tf.zeros(shape=(batch_size, input_caps, caps, 1, 1))  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
        iteration = 0
        # update logits at each routing iteration
        tf.while_loop(
            cond=lambda i, l, a: i < self.routing_iter,
            body=routing_loop,
            loop_vars=[iteration, logits, pre_activation],
            back_prop=False
        )
        # return activation from the updated logits
        return routing_step(logits, pre_activation)
