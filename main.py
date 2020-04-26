#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras as k
from tensorflow.keras.datasets import mnist

from capsnet import CapsConv2D, CapsDense, Losses, Metrics, CapsConv

# Set random seeds so that the same outputs are generated always
np.random.seed(42)
tf.random.set_seed(42)


def safe_l2_norm(_data, axis=-1, keepdims=False):
    squared_norm = tf.reduce_sum(tf.square(_data), axis=axis, keepdims=keepdims)
    return tf.sqrt(squared_norm + k.backend.epsilon())


def max_mask(_data):
    """
    Mask data from all capsules except the most activated one, for each instance
    :param _data: shape: (None, num_caps, dim_caps)
    :return:
    """
    _norm = safe_l2_norm(_data, axis=-1)  # shape: (None, num_caps)
    _y_pred = tf.argmax(_norm, axis=-1)  # shape: (None, )
    _mask = tf.expand_dims(tf.one_hot(_y_pred, depth=_norm.shape[-1]), axis=-1)  # shape: (None, num_caps, 1)
    _masked = tf.multiply(_data, _mask)  # shape: (None, num_caps, dim_caps)
    return _masked


def create_capsnet_model(input_shape, name) -> k.Model:
    # input
    l1 = k.layers.Input(shape=input_shape, name='input')  # type: tf.Tensor
    # initial convolution
    l2 = k.layers.Conv2D(filters=256, kernel_size=(9, 9), strides=(1, 1), activation='relu', name='conv')(l1)  # type: tf.Tensor
    # layer to convert to capsule domain
    l3 = CapsConv2D(caps_filters=32, caps_dims=8, kernel_size=(9, 9), strides=(2, 2), activation='relu', name='conv_caps_2d')(l2)  # type: tf.Tensor
    # conv capsule layer with dynamic routing
    l3 = CapsConv(caps_filters=32, caps_dims=8, kernel_size=(2, 2), strides=(1, 1), routing_iter=3, name='conv_caps_3d')(l3)  # type: tf.Tensor
    # dense capsule layer with dynamic routing
    l4 = CapsDense(caps=10, caps_dims=16, routing_iter=3, name='dense_caps')(l3)  # type: tf.Tensor
    # decoder
    d0 = k.layers.Lambda(max_mask, name="masking")(l4)  # type: tf.Tensor
    d1 = k.layers.Flatten(name="flatten")(d0)  # type: k.layers.Layer
    d2 = k.layers.Dense(512, activation='relu', name="decoder_l1")(d1)  # type: tf.Tensor
    d3 = k.layers.Dense(1024, activation='relu', name="decoder_l2")(d2)  # type: tf.Tensor
    d4 = k.layers.Dense(tf.reduce_prod(input_shape), activation='sigmoid', name="decoder_l3")(d3)  # type: tf.Tensor
    # output layers
    margin = k.layers.Lambda(safe_l2_norm, name='margin')(l4)  # type: tf.Tensor
    reconstruction = k.layers.Reshape(input_shape, name='reconstruction')(d4)  # type: tf.Tensor
    # define the model
    return k.models.Model(inputs=l1, outputs=[margin, reconstruction], name=name)


if __name__ == '__main__':
    NUM_CLASSES = 10
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # one hot encode data
    x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]
    y_train, y_test = k.utils.to_categorical(y_train, NUM_CLASSES), k.utils.to_categorical(y_test, NUM_CLASSES)

    model = create_capsnet_model(input_shape=x_train.shape[1:], name='mnist_capsnet')
    model.compile(optimizer='adam', loss=[Losses.margin_loss, Losses.reconstruction_loss], loss_weights=[1e0, 5e-3],
                  metrics={'margin': Metrics.accuracy})
    model.summary()

    # checkpoint function to save best weights
    checkpoint = k.callbacks.ModelCheckpoint("best_weights.hdf5", save_best_only=True)

    if os.path.exists('best_weights.hdf5'):
        # load existing weights
        model.load_weights('best_weights.hdf5')
        # evaluation
        model.evaluate(x_test, [y_test, x_test])
    else:
        # training
        model.fit(x_train, [y_train, x_train], batch_size=50, epochs=5, validation_split=0.1, callbacks=[checkpoint])


    def print_results():
        indices = np.random.randint(0, len(x_test), 10)
        _n, _x, _y = len(indices), x_test[indices], y_test[indices]
        [_y_p, _x_p] = model.predict(_x)
        fig, axs = plt.subplots(ncols=5, nrows=4)
        for z in range(_n):
            i = (z // 5) * 2
            j = z % 5
            axs[i, j].imshow(np.squeeze(_x_p[z]), cmap='gray', vmin=0.0, vmax=1.0)
            axs[i, j].axis('off')
            axs[i + 1, j].imshow(np.squeeze(_x[z]), cmap='gray', vmin=0.0, vmax=1.0)
            axs[i + 1, j].axis('off')
        fig.show()


    print_results()
