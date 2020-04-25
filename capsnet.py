import tensorflow as tf
from tensorflow import keras as k
from tensorflow_core.python.keras.utils import conv_utils


class Activations:
    @staticmethod
    def squash(_data, axis=-1):
        """
        Normalize to unit vectors
        :param _data: Tensor with rank >= 2
        :param axis:
        :return:
        """
        square_sum = tf.reduce_sum(tf.square(_data), axis=axis, keepdims=True)
        squash_factor = square_sum / (1. + square_sum)
        unit_vector = _data / tf.sqrt(square_sum + k.backend.epsilon())
        return squash_factor * unit_vector


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
        result = tf.reshape(result, shape=(-1, result.shape[1], result.shape[2], result.shape[3] // self.caps_dims, self.caps_dims))
        return Activations.squash(result, axis=-1)


class CapsConv(k.layers.Conv3D):
    def __init__(self, caps_layers, caps_dims, routing_iter, kernel_size, strides, **kwargs):
        self.caps_layers = caps_layers
        self.caps_dims = caps_dims
        self.routing_iter = routing_iter
        self.w = ...
        # only accept kernel_size and strides specified in 2D
        kernel_size = conv_utils.normalize_tuple(kernel_size, 2, name='kernel_size')
        strides = conv_utils.normalize_tuple(strides, 2, name='strides')
        super(CapsConv, self).__init__(self.caps_layers * self.caps_dims, (*kernel_size, 1), strides=(*strides, 1), **kwargs)

    def build(self, input_shape: tf.TensorShape):
        assert input_shape.rank == 5
        input_caps_dims = input_shape[4]
        # configure kernel_size and stride for capsule-wise convolution in the last dimension
        self.kernel_size = (*self.kernel_size[:-1], input_caps_dims)
        self.strides = (*self.strides[:-1], input_caps_dims)
        super(CapsConv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # reshape (..., caps_layers, caps_dims) for depth-wise convolution
        new_shape = (*inputs.shape[:-2], inputs.shape[-2] * inputs.shape[-1], 1)
        inputs = tf.reshape(inputs, shape=new_shape)  # shape: (batch_size, rows, cols, input_caps_layers * input_caps_dims, 1)
        # perform 3D convolution
        result = super(CapsConv, self).call(inputs)  # shape: (batch_size, new_rows, new_cols, input_caps_layers, caps_layers * caps_dims)
        # transpose and reshape
        result = tf.linalg.matrix_transpose(result)  # shape: (batch_size, new_rows, new_cols, caps_layers * caps_dims, input_caps_layers)
        new_shape = (-1, result.shape[1], result.shape[2], result.shape[3] // self.caps_dims, self.caps_dims, result.shape[4])
        result = tf.reshape(result, shape=new_shape)  # shape: (batch_size, new_rows, new_cols, caps_layers, caps_dims, input_caps_layers)
        # dynamic routing
        return Activations.squash(result, axis=-1)


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
        assert input_shape.rank == 5
        self.input_caps = input_shape[1] * input_shape[2] * input_shape[3]
        self.input_caps_dims = input_shape[4]
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
        def routing_step(_routing_weights, _prediction):
            """
            Weight the prediction by routing weights, squash it, and return it
            :param _routing_weights: (batch_size, p_num_caps, num_caps, 1, 1)
            :param _prediction: (batch_size, p_num_caps, num_caps, dim_caps, 1)
            :return:
            """
            # softmax of weights over num_caps axis
            prob_w = tf.nn.softmax(_routing_weights, axis=2)  # shape: (batch_size, p_num_caps, num_caps, 1, 1)
            # elementwise multiplication of weights with prediction
            w_pred = tf.multiply(prob_w, _prediction)  # shape: (batch_size, p_num_caps, num_caps, dim_caps, 1)
            # sum over p_num_caps axis
            w_prediction_sum = tf.reduce_sum(w_pred, axis=1, keepdims=True)  # shape: (batch_size, 1, num_caps, dim_caps, 1)
            # squash over dim_caps and return
            return Activations.squash(w_prediction_sum, axis=-2)  # shape: (batch_size, 1, num_caps, dim_caps, 1)

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
