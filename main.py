#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras as k
from tensorflow.keras.datasets import mnist

from capsnet import Losses, Metrics, NN, StackedConvCaps, ConvCaps, FlattenCaps

# Set random seeds so that the same outputs are generated always
np.random.seed(42)
tf.random.set_seed(42)


def max_mask(inputs):
    """
    Mask data from all capsules except the most activated one, for each instance
    :param inputs: shape: (None, num_caps, dim_caps)
    :return:
    """
    norm = NN.norm(inputs, axis=-1)  # shape: (None, num_caps)
    argmax = tf.argmax(norm, axis=-1)  # shape: (None, )
    mask = tf.expand_dims(tf.one_hot(argmax, depth=norm.shape[-1]), axis=-1)  # shape: (None, num_caps, 1)
    masked_input = tf.multiply(inputs, mask)  # shape: (None, num_caps, dim_caps)
    return masked_input


def mask_cid(inputs):
    """
    Select most activated capsule from each instance and return it
    :param inputs: shape: (None, num_caps, dim_caps)
    :return:
    """
    norm = NN.norm(inputs, axis=-1)  # shape: (None, num_caps)
    # build index of elements to collect
    i = tf.range(start=0, limit=tf.shape(inputs)[0], delta=1)  # shape: (None, )
    j = tf.argmax(norm, axis=-1)  # shape: (None, )
    ij = tf.stack([i, tf.cast(j, tf.int32)], axis=1)
    # gather from index and return
    return tf.gather_nd(inputs, ij)


def fc_decoder(input_shape, target_shape, name):
    il = k.layers.Input(shape=input_shape, name='input')
    hl = k.layers.Lambda(max_mask, name="masking")(il)
    hl = k.layers.Flatten(name="flatten")(hl)
    hl = k.layers.Dense(512, activation='relu', name="decoder_l1")(hl)
    hl = k.layers.Dense(1024, activation='relu', name="decoder_l2")(hl)
    ol = k.layers.Dense(tf.reduce_prod(target_shape), activation='sigmoid', name="decoder_l3")(hl)
    return k.models.Model(inputs=il, outputs=ol, name=name)


def conv_decoder(input_shape, target_shape, name):
    il = k.layers.Input(shape=input_shape, name='input')
    dl = k.layers.Lambda(mask_cid)(il)
    dl = k.layers.Dense(11 * 11 * 256)(dl)
    dl = k.layers.Reshape((11, 11, 256))(dl)
    dl = k.layers.BatchNormalization(momentum=0.8)(dl)
    dl = k.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu')(dl)
    ol = k.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), output_padding=(1, 1), activation='relu')(dl)
    return k.models.Model(inputs=il, outputs=ol, name=name)


def create_capsnet_model(input_shape, name) -> k.Model:
    # input layer
    il = k.layers.Input(shape=input_shape, name='input')
    # encoder
    cl = k.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv')(il)
    # capsule block 1
    cap1 = ConvCaps(filters=32, filter_dims=4, kernel_size=(3, 3), strides=(1, 1), name='cap1_l1')(cl)
    cap1 = StackedConvCaps(filters=32, filter_dims=8, routing_iter=0, kernel_size=(3, 3), strides=(1, 1), name='cap1_l2')(cap1)
    cap1 = StackedConvCaps(filters=32, filter_dims=16, routing_iter=0, kernel_size=(3, 3), strides=(1, 1), name='cap1_l3')(cap1)
    # merging
    # fl = DenseCaps(caps=10, caps_dims=16, routing_iter=3, name='prediction')(cap1)
    fl = FlattenCaps(caps=10, name='prediction')(cap1)
    # decoder
    decoder = fc_decoder(input_shape=fl.shape[1:], target_shape=input_shape, name="fc_decoder")(fl)
    # decoder = conv_decoder(input_shape=hl.shape[1:], target_shape=input_shape, name="conv_decoder")(hl)

    # output layers
    margin = k.layers.Lambda(NN.norm, name='margin')(fl)
    reconstruction = k.layers.Reshape(input_shape, name='reconstruction')(decoder)
    # define the model
    return k.models.Model(inputs=il, outputs=[margin, reconstruction], name=name)


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
    model.compile(optimizer='adam', loss=[Losses.margin_loss, Losses.reconstruction_loss], loss_weights=[1, 0],
                  metrics={'margin': Metrics.accuracy})
    model.summary(line_length=120)

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
