import tensorflow as tf
from tensorflow import keras as k


class NN:
    @staticmethod
    def norm(data, e=k.backend.epsilon(), axis=-1, keepdims=False):
        squared_norm = tf.reduce_sum(tf.square(data), axis=axis, keepdims=keepdims)
        return tf.sqrt(squared_norm + e)

    @staticmethod
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


class Losses:
    @staticmethod
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

    @staticmethod
    def reconstruction_loss(_y_true, _y_pred):
        """
        Mean Squared Error

        :param _y_true: shape: (None, 28, 28, 1)
        :param _y_pred: shape: (None, 28, 28, 1)
        :return:
        """
        return tf.reduce_mean(tf.square(_y_true - _y_pred))


class Metrics:
    @staticmethod
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


class ConvCaps(k.layers.Conv2D):
    def __init__(self, filters, filter_dims, kernel_size, **kwargs):
        self.filters = filters
        self.filter_dims = filter_dims
        super(ConvCaps, self).__init__(self.filters * self.filter_dims, kernel_size, **kwargs)

    def call(self, inputs, **kwargs):
        result = super(ConvCaps, self).call(inputs)
        result = tf.reshape(result, shape=(-1, *result.shape[1:3], result.shape[3] // self.filter_dims, self.filter_dims))
        activation = NN.squash(result, axis=-1)
        return activation


class StackedConvCaps(k.layers.Layer):
    def __init__(self, filters, filter_dims, routing_iter, kernel_size, strides, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.filter_dims = filter_dims
        self.routing_iter = routing_iter
        # build-time parameters
        self.input_filters = ...
        self.input_filter_dims = ...
        self.conv_layer = ...  # type: k.layers.Conv3D
        # only accept kernel_size and strides specified in 2D
        self.kernel_size = kernel_size
        self.strides = strides

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
            strides=(*self.strides, self.input_filter_dims)
        )
        # mark as built
        self.built = True

    def call(self, inputs, **kwargs):
        # reshape (batch_size), (input_rows, input_cols, input_caps_filters, input_caps_dims) for capsule-wise convolution
        s = (-1, *inputs.shape[1:3], self.input_filters * self.input_filter_dims, 1)
        inputs = tf.reshape(inputs, shape=s)  # shape: (batch_size), (input_rows, input_cols, input_filters * input_filter_dims, 1)
        # perform 3D convolution
        result = self.conv_layer(inputs)  # shape: (b,p,q,s,r * n)
        # reshape into (b,p,q,s,r,n)
        result = tf.reshape(result, shape=(-1, *result.shape[1:4], self.filters, self.filter_dims))
        # transpose into (b,p,q,r,s,n)
        result = tf.transpose(result, perm=(0, 1, 2, 4, 3, 5))
        # get activation by dynamic routing
        activation = self.dynamic_routing(pre_activation=result)  # shape: (b,p,q,r,1,n)
        # return activation in (b,p,q,r,n) form
        return tf.squeeze(activation, axis=-2)

    def dynamic_routing(self, pre_activation):
        """
        Dynamic routing in 3D Convolution.

        Terminology Used:
        batch_size      (b),
        rows            (p),
        cols            (q),
        filter_dims     (n),
        filters         (r),
        input_filters   (s),

        :param pre_activation (b,p,q,r,s,n)
        :return: activation (b,p,q,r,1,n)
        """

        @tf.function
        def softmax(_logits, axis):
            return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True)

        @tf.function
        def routing_step(_logits, _pre_activation):
            # softmax of logits over 3D space (such that their sum is 1)
            _prob = softmax(_logits, axis=(1, 2, 3))  # shape: (b,p,q,r,s,1)
            # calculate activation based on _prob
            _activation = tf.reduce_sum(_prob * _pre_activation, axis=-2, keepdims=True)  # shape: (b,p,q,r,1,n)
            # squash over 3D space and return
            return NN.squash(_activation, axis=(1, 2, 3))  # shape: (b,p,q,r,1,n)

        @tf.function
        def routing_loop(_i, _logits, _pre_activation):
            # step 1: find the activation from logits
            _activation = routing_step(_logits, _pre_activation)  # shape: (b,p,q,r,1,n)
            # step 2: find the agreement (dot product) between pre_activation (b,p,q,r,s,n) and activation (b,p,q,r,1,n), across dim_caps
            _agreement = tf.reduce_sum(_pre_activation * _activation, axis=-1, keepdims=True)  # shape: (b,p,q,r,s,1)
            # update routing weight
            _logits = _logits + _agreement
            # return updated variables
            return _i + 1, _logits, _pre_activation

        # define dimensions
        b = tf.shape(pre_activation)[0]
        [p, q, r, s, _] = pre_activation.shape[1:]
        # define variables
        logits = tf.zeros(shape=(b, p, q, r, s, 1), dtype=tf.float32)  # shape: (b,p,q,r,s,1)
        i = 0
        # update logits at each routing iteration
        tf.while_loop(
            cond=lambda _i, _logits, _pre_activation: i < self.routing_iter,
            body=routing_loop,
            loop_vars=[i, logits, pre_activation],
            swap_memory=True
        )
        # return activation from the updated logits
        return routing_step(logits, pre_activation)  # shape: (b,p,q,r,1,n)


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
            shape=(1, self.input_caps, self.caps, self.caps_dims, self.input_caps_dims),
            dtype=tf.float32,
            initializer='random_normal'
        )
        self.built = True

    def call(self, inputs, **kwargs):
        # get batch size of input
        batch_size = tf.shape(inputs)[0]
        # reshape input
        inputs = tf.reshape(inputs, (batch_size, self.input_caps, 1, self.input_caps_dims, 1))  # shape: (batch_size, p_num_caps, 1. p_dim_caps, 1)
        inputs = tf.tile(inputs, [1, 1, self.caps, 1, 1])  # shape: (batch_size, p_num_caps, num_caps, p_dim_caps, 1)
        # tile transformation matrix for each element in batch
        w = tf.tile(self.w, [batch_size, 1, 1, 1, 1])  # shape: (batch_size, p_num_caps, num_caps, dim_caps, p_dim_caps)
        # calculate prediction (dot product of the last two dimensions of tiled_w and input)
        result = tf.matmul(w, inputs)  # shape: (batch_size, p_num_caps, num_caps, dim_caps, 1)
        # dynamic routing
        activation = self.dynamic_routing(pre_activation=result)  # shape: (batch_size, 1, num_caps, dim_caps, 1)
        # reshape to (None, num_caps, dim_caps) and return
        return tf.reshape(activation, shape=(-1, self.caps, self.caps_dims))

    def dynamic_routing(self, pre_activation):
        """
        Dynamic Routing as proposed in the original paper

        :param pre_activation: shape: (batch_size, p_num_caps, num_caps, dim_caps, 1)
        :return:
        """

        @tf.function
        def routing_step(_logits, _pre_activation):
            """
            Weight the prediction by routing weights, squash it, and return it
            :param _logits: (batch_size, p_num_caps, num_caps, 1, 1)
            :param _pre_activation: (batch_size, p_num_caps, num_caps, dim_caps, 1)
            :return:
            """
            # softmax of logits over all capsules (such that their sum is 1)
            _prob = tf.nn.softmax(_logits, axis=2)  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
            # calculate activation based on _prob
            _activation = tf.reduce_sum(_prob * _pre_activation, axis=1, keepdims=True)  # shape: (batch_size, 1, num_caps, dim_caps, 1)
            # squash over dim_caps and return
            return NN.squash(_activation, axis=-2)  # shape: (batch_size, 1, num_caps, dim_caps, 1)

        @tf.function
        def routing_loop(_i, _logits, _pre_activation):
            # step 1: find the activation from logits
            _activation = routing_step(_logits, _pre_activation)  # shape: (batch_size, 1, num_caps, dim_caps, 1)
            # step 2: find the agreement (dot product) between pre_activation and activation, across dim_caps
            _agreement = tf.reduce_sum(_pre_activation * _activation, axis=-2, keepdims=True)  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
            # step 3: update routing weights based on agreement
            _logits = _logits + _agreement
            # return updated variables
            return _i + 1, _logits, _pre_activation

        tensor_shape = tf.shape(pre_activation)
        batch_size = tensor_shape[0]
        input_caps = tensor_shape[1]
        caps = tensor_shape[2]
        # define variables
        logits = tf.zeros(shape=(batch_size, input_caps, caps, 1, 1), dtype=tf.float32)  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
        i = 0
        # update logits at each routing iteration
        tf.while_loop(
            cond=lambda _i, _logits, _pre_activation: i < self.routing_iter,
            body=routing_loop,
            loop_vars=[i, logits, pre_activation],
            swap_memory=True
        )
        # return activation from the updated logits
        return routing_step(logits, pre_activation)


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
            shape=(self.input_caps, self.caps),
            dtype=tf.float32,
            initializer='random_normal'
        )
        self.built = True

    def call(self, inputs, **kwargs):
        inputs = tf.reshape(inputs, (-1, self.input_caps, self.input_caps_dims))
        return tf.einsum('bcd,cy->byd', inputs, self.w)
