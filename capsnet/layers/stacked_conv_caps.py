import tensorflow as tf
from tensorflow import keras as k

from capsnet.nn import softmax, squash


@tf.function(input_signature=(
        tf.TensorSpec(shape=(None, None, None, None, None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None, None, None, None), dtype=tf.float32),
))
def routing_step(_logits, _pre_activation):
    # softmax of logits over 3D space (such that their sum is 1)
    _prob = softmax(_logits, axis=tf.constant([1, 2, 3]))  # shape: (b,p,q,r,s,1)
    # calculate activation based on _prob
    _activation = tf.reduce_sum(_prob * _pre_activation, axis=-2, keepdims=True)  # shape: (b,p,q,r,1,n)
    # return _activation  # temporary hack to get the gradients flowing
    # squash over 3D space and return
    return squash(_activation, axis=tf.constant([-1]))  # shape: (b,p,q,r,1,n)


@tf.function(input_signature=(
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None, None, None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None, None, None, None), dtype=tf.float32),
))
def routing_loop(_i, _logits, _pre_activation):
    # step 1: find the activation from logits
    _activation = routing_step(_logits, _pre_activation)  # shape: (b,p,q,r,1,n)
    # step 2: find the agreement (dot product) between pre_activation (b,p,q,r,s,n) and activation (b,p,q,r,1,n), across dim_caps
    _agreement = tf.reduce_sum(_pre_activation * _activation, axis=-1, keepdims=True)  # shape: (b,p,q,r,s,1)
    # update routing weight
    _logits = _logits + _agreement
    # return updated variables
    return _i + 1, _logits, _pre_activation


class StackedConvCaps(k.layers.Layer):
    def __init__(self, filters, filter_dims, routing_iter, kernel_size, strides, padding='valid', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.filter_dims = filter_dims
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.routing_iter = routing_iter
        # build-time parameters
        self.input_filters = ...
        self.input_filter_dims = ...
        self.conv_layer = ...  # type: k.layers.Conv3D

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'filter_dims': self.filter_dims,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'routing_iter': self.routing_iter,
            'input_filers': self.input_filters,
            'input_filter_dims': self.input_filter_dims
        })
        return config

    def build(self, input_shape: tf.TensorShape):
        # sanity check
        tf.assert_equal(input_shape.rank, 5, message=f'Expected Tensor of Rank = 5, Got Rank = {input_shape.rank}')
        # define capsule parameters
        self.input_filters = input_shape[3]
        self.input_filter_dims = input_shape[4]
        # configure convolution layer for capsule-wise convolution in the last dimension
        self.conv_layer = k.layers.Conv3D(
            filters=self.filters * self.filter_dims,
            kernel_size=(*self.kernel_size, self.input_filter_dims),
            strides=(*self.strides, self.input_filter_dims),
            padding=self.padding
        )
        # mark as built
        self.built = True

    def call(self, inputs, **kwargs):
        # reshape (batch_size), (input_rows, input_cols, input_caps_filters, input_caps_dims) for capsule-wise convolution
        s = (-1, *inputs.shape[1:3], self.input_filters * self.input_filter_dims, 1)
        inputs = tf.reshape(inputs, shape=s)  # shape: (batch_size), (input_rows, input_cols, input_filters * input_filter_dims, 1)
        # perform 3D convolution
        initial_activation = self.conv_layer(inputs)  # shape: (b,p,q,s,r * n)
        # reshape into (b,p,q,s,r,n)
        initial_activation = tf.reshape(initial_activation, shape=(-1, *initial_activation.shape[1:4], self.filters, self.filter_dims))
        # transpose into (b,p,q,r,s,n)
        initial_activation = tf.transpose(initial_activation, perm=(0, 1, 2, 4, 3, 5))
        # get activation by dynamic routing
        activation = self.dynamic_routing(initial_activation)  # shape: (b,p,q,r,1,n)
        # return activation in (b,p,q,r,n) form
        return tf.squeeze(activation, axis=-2)

    def dynamic_routing(self, initial_activation):
        """
        Dynamic routing in 3D Convolution.

        Terminology Used:
        batch_size      (b),
        rows            (p),
        cols            (q),
        filter_dims     (n),
        filters         (r),
        input_filters   (s),

        :param initial_activation (b,p,q,r,s,n)
        :return: activation (b,p,q,r,1,n)
        """
        # define dimensions
        b = tf.shape(initial_activation)[0]
        [p, q, r, s, _] = initial_activation.shape[1:]
        # define variables
        initial_logits = tf.zeros(shape=(b, p, q, r, s, 1))  # shape: (b,p,q,r,s,1)
        # update logits at each routing iteration
        [_, final_logits, _] = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(
            loop_vars=[tf.constant(0), initial_logits, initial_activation],
            cond=lambda i, l, a: i < self.routing_iter,
            body=routing_loop
        ))
        # return activation from the updated logits
        return routing_step(final_logits, initial_activation)  # shape: (b,p,q,r,1,n)
