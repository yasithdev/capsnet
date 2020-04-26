import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.layers import InputSpec
from tensorflow_core.python.keras.utils import conv_utils


class NN:
    @staticmethod
    def norm(data, e=k.backend.epsilon(), axis=-1, keepdims=False):
        squared_norm = tf.reduce_sum(tf.square(data), axis=axis, keepdims=keepdims)
        return tf.sqrt(squared_norm + e)

    @staticmethod
    def squash(data, axis=-1):
        """
        Normalize to unit vectors
        :param data: Tensor with rank >= 2
        :param axis: axis over which to squash
        :return:
        """
        e = k.backend.epsilon()
        squared_norm = tf.reduce_sum(tf.square(data), axis=axis, keepdims=True)
        return (squared_norm / (1 + squared_norm)) * (data / tf.sqrt(squared_norm + e))

    @staticmethod
    def softmax(logits: tf.Tensor, axis) -> tf.Tensor:
        """
        softmax over multiple dimensions
        :param axis: axis/axes over which to normalize
        :param logits: logits tensor
        """
        return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis=axis, keepdims=True)


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


class CapsConv2D(k.layers.Conv2D):
    def __init__(self, caps_filters, caps_dims, kernel_size, **kwargs):
        self.caps_filters = caps_filters
        self.caps_dims = caps_dims
        super(CapsConv2D, self).__init__(self.caps_filters * self.caps_dims, kernel_size, **kwargs)

    def call(self, inputs, **kwargs):
        result = super(CapsConv2D, self).call(inputs)
        result = tf.reshape(result, shape=(-1, *result.shape[1:3], result.shape[3] // self.caps_dims, self.caps_dims))
        activation = NN.squash(result, axis=-1)
        return activation


class CapsConv3D(k.layers.Conv3D):
    def __init__(self, caps_filters, caps_dims, routing_iter, kernel_size, strides, **kwargs):
        self.caps_filters = caps_filters
        self.caps_dims = caps_dims
        self.routing_iter = routing_iter
        # build-time parameters
        self.input_caps_filters = ...
        self.input_caps_dims = ...
        self.conv_input_shape = ...
        self.w = ...
        # only accept kernel_size and strides specified in 2D
        kernel_size = conv_utils.normalize_tuple(kernel_size, 2, name='kernel_size')
        strides = conv_utils.normalize_tuple(strides, 2, name='strides')
        # call super init function
        super(CapsConv3D, self).__init__(self.caps_filters * self.caps_dims, (*kernel_size, 1), strides=(*strides, 1), **kwargs)

    def build(self, input_shape: tf.TensorShape):
        # sanity check
        tf.assert_equal(input_shape.rank, 5, message=f'Expected Tensor of Rank = 5, Got Rank = {input_shape.rank}')
        # define capsule parameters
        self.input_caps_filters = input_shape[3]
        self.input_caps_dims = input_shape[4]
        # configure kernel_size and stride for capsule-wise convolution in the last dimension
        self.kernel_size = (*self.kernel_size[:-1], self.input_caps_dims)
        self.strides = (*self.strides[:-1], self.input_caps_dims)
        # call super build function. pass input shape for 3D convolution here
        super(CapsConv3D, self).build((*input_shape[:3], self.input_caps_filters * self.input_caps_dims, 1))
        # update input spec to match original rank
        self.input_spec = InputSpec(ndim=5)

    def call(self, inputs, **kwargs):
        # reshape (batch_size), (input_rows, input_cols, input_caps_filters, input_caps_dims) for capsule-wise convolution
        new_shape = (-1, *inputs.shape[1:3], self.input_caps_filters * self.input_caps_dims, 1)
        inputs = tf.reshape(inputs, shape=new_shape)  # shape: (batch_size), (input_rows, input_cols, input_caps_filters * input_caps_dims, 1)
        # perform 3D convolution
        result = super(CapsConv3D, self).call(inputs)  # shape: (batch_size), (rows, cols, input_caps_filters, caps_filters * caps_dims)
        # reshape into (batch_size), (rows, cols, input_caps_filters, caps_filters, caps_dims)
        result = tf.reshape(result, shape=(-1, *result.shape[1:4], self.caps_filters, self.caps_dims))
        # activation by dynamic routing
        activation = CapsConv3D.dynamic_routing(
            routing_iter=self.routing_iter,
            caps_filters=self.caps_filters,
            input_caps_filters=self.input_caps_filters,
            prediction=result
        )  # shape: (batch_size), (rows(p), cols(q), caps_filters(r), caps_dims(t))
        # return result
        return activation

    @staticmethod
    def dynamic_routing(routing_iter, caps_filters, input_caps_filters, prediction):
        """
        Dynamic routing

        :param input_caps_filters:
        :param caps_filters:
        :param routing_iter: number of routing iterations, >=1
        :param prediction: shape: (batch_size(b)), (rows(p), cols(q), input_caps_filters(s), caps_filters(r), caps_dims(t))
        :return:
        """
        # sanity check
        tf.assert_greater(routing_iter, 0, message='Dynamic Routing should have >= 1 iterations')
        # get batch size
        batch_size = tf.shape(prediction)[0]
        # define routing weight - shape: (batch_size(b)), (rows(p), cols(q), caps_filters(r), input_caps_filters(s))
        logits = tf.zeros(shape=(batch_size, *prediction.shape[1:3], caps_filters, input_caps_filters))
        # placeholder for activation
        activation = ...
        # routing
        for _ in range(routing_iter):
            # calculate coupling coefficient - shape: (batch_size(b)), (rows(p), cols(q), caps_filters(r), input_caps_filters(s))
            cc = NN.softmax(logits, axis=(1, 2, 3))
            # calculate routed prediction - shape: (batch_size), (rows(p), cols(q), caps_filters(r), caps_dims(t))
            routed_prediction = tf.einsum('bpqrs,bpqsrt->bpqrt', cc, prediction)
            # squash over caps_dims - shape: (batch_size), (rows(p), cols(q), caps_filters(r), caps_dims(t))
            activation = NN.squash(routed_prediction, axis=-1)
            # calculate agreement between original prediction and routed prediction
            agreement = tf.einsum('bpqrt,bpqsrt->bpqrs', activation, prediction)
            # update routing weight
            logits = logits + agreement
        # return routed activation
        return activation


class CapsDense(k.layers.Layer):
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

    @staticmethod
    @tf.function
    def dynamic_routing(routing_iter, batch_size, input_caps, caps, prediction):
        def routing_step(logits, _prediction):
            """
            Weight the prediction by routing weights, squash it, and return it
            :param logits: (batch_size, p_num_caps, num_caps, 1, 1)
            :param _prediction: (batch_size, p_num_caps, num_caps, dim_caps, 1)
            :return:
            """
            # softmax of weights over num_caps axis
            prob_w = NN.softmax(logits, axis=2)  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
            # elementwise multiplication of weights with prediction
            w_pred = tf.multiply(prob_w, _prediction)  # shape: (batch_size, p_num_caps, num_caps, dim_caps, 1)
            # sum over p_num_caps axis
            w_prediction_sum = tf.reduce_sum(w_pred, axis=1, keepdims=True)  # shape: (batch_size, 1, num_caps, dim_caps, 1)
            # squash over dim_caps and return
            return NN.squash(w_prediction_sum, axis=-2)  # shape: (batch_size, 1, num_caps, dim_caps, 1)

        # initialize routing weights to zero
        routing_weights = tf.zeros(shape=(batch_size, input_caps, caps, 1, 1), dtype=tf.float32)  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
        # initial routed prediction
        routed_prediction = routing_step(routing_weights, prediction)  # shape: (batch_size, 1, num_caps, dim_caps, 1)
        # update routing weights and routed prediction, for routing_iter iterations
        for i in range(routing_iter):
            # step 1: tile the weighted prediction for each previous capsule
            tiled = tf.tile(routed_prediction, [1, input_caps, 1, 1, 1])  # shape: (batch_size, p_num_caps, num_caps, dim_caps, 1)
            # step 2: find the agreement between prediction and weighted prediction
            agreement = tf.matmul(prediction, tiled, transpose_a=True)  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
            # step 3: update routing weights based on agreement
            routing_weights = tf.add(routing_weights, agreement)
            # step 4: update routed prediction
            routed_prediction = routing_step(routing_weights, prediction)  # shape: (batch_size, 1, num_caps, dim_caps, 1)
        # return final routed prediction
        return routed_prediction

    def call(self, inputs, **kwargs):
        # get batch size of input
        batch_size = tf.shape(inputs)[0]
        # reshape input
        inputs = tf.reshape(inputs, (batch_size, self.input_caps, self.input_caps_dims))  # shape: (batch_size, p_num_caps, p_dim_caps)
        inputs = tf.expand_dims(inputs, axis=-1)  # shape: (batch_size, p_num_caps, p_dim_caps, 1)
        inputs = tf.expand_dims(inputs, axis=2)  # shape: (batch_size, p_num_caps, 1, p_dim_caps, 1)
        inputs = tf.tile(inputs, [1, 1, self.caps, 1, 1])  # shape: (batch_size, p_num_caps, num_caps, p_dim_caps, 1)
        # tile transformation matrix for each element in batch
        tiled_w = tf.tile(self.w, [batch_size, 1, 1, 1, 1])  # shape: (batch_size, p_num_caps, num_caps, dim_caps, p_dim_caps)
        # calculate prediction (dot product of the last two dimensions of tiled_w and input)
        prediction = tf.matmul(tiled_w, inputs)  # shape: (batch_size, p_num_caps, num_caps, dim_caps, 1)
        # dynamic routing
        routed_prediction = CapsDense.dynamic_routing(
            routing_iter=self.routing_iter,
            batch_size=batch_size,
            input_caps=self.input_caps,
            caps=self.caps,
            prediction=prediction
        )  # shape: (batch_size, 1, num_caps, dim_caps, 1)
        # reshape to (None, num_caps, dim_caps) and return
        return tf.reshape(routed_prediction, shape=(-1, self.caps, self.caps_dims))


class CapsFlatten(k.layers.Layer):
    def __init__(self, caps, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(CapsFlatten, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.caps = caps
        self.input_caps = ...
        self.input_caps_dims = ...
        self.w = ...

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
