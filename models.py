#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers as kl

import functions
from capsnet import nn, layers
from capsnet.layers import ConvCaps, DenseCaps


def get_model(name, input_shape, num_classes) -> k.Model:
    if name == "original":
        return original_model(name, input_shape, num_classes)
    elif name == "deepcaps":
        return deep_caps_model(name, input_shape, num_classes)
    else:
        exit(1)


def original_model(name, input_shape, num_classes) -> k.Model:
    inl = kl.Input(shape=input_shape, name='input')
    nl = kl.Conv2D(filters=256, kernel_size=(9, 9), strides=(1, 1), activation='relu', name='conv')(inl)
    nl = ConvCaps(filters=32, filter_dims=8, kernel_size=(9, 9), strides=(2, 2), activation='relu', name='conv_caps_2d')(nl)
    nl = DenseCaps(caps=num_classes, caps_dims=16, routing_iter=3, name='dense_caps')(nl)
    pred = kl.Lambda(nn.norm, name='pred')(nl)
    recon = fully_connected_decoder(input_shape)(nl)
    return k.Model(inputs=inl, outputs=[pred, recon], name=name)


def fully_connected_decoder(target_shape):
    def decoder(input_tensor):
        nl = kl.Lambda(functions.mask, name="decoder_masking")(input_tensor)
        nl = kl.Flatten(name="decoder_flatten")(nl)
        nl = kl.Dense(512, activation='relu', name="decoder_dense_1")(nl)
        nl = kl.Dense(1024, activation='relu', name="decoder_dense_2")(nl)
        nl = kl.Dense(tf.reduce_prod(target_shape), activation='sigmoid', name="decoder_dense_3")(nl)
        nl = kl.Reshape(target_shape, name='recon')(nl)
        return nl

    return decoder


def deep_caps_model(name, input_shape, num_classes) -> k.Model:
    inl = k.layers.Input(shape=input_shape, name='input')
    nl = k.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation='relu', name='conv2d')(inl)
    nl = layers.ConvCaps(filters=32, filter_dims=4, kernel_size=(3, 3), strides=(1, 1), name='cap1_conv1')(nl)
    nl = layers.StackedConvCaps(filters=32, filter_dims=8, routing_iter=3, kernel_size=(3, 3), strides=(1, 1), name='cap1_conv2')(nl)
    nl = layers.FlattenCaps(caps=num_classes, name='cap1_flatten')(nl)
    pred = k.layers.Lambda(nn.norm, name='pred')(nl)
    recon = fully_connected_decoder(target_shape=input_shape)(nl)
    return k.models.Model(inputs=inl, outputs=[pred, recon], name=name)


def conv_decoder(target_shape):
    def decoder(input_tensor):
        nl = kl.Lambda(functions.mask_cid, name="decoder_masking")(input_tensor)
        nl = kl.Dense(11 * 11 * 256, name="decoder_dense")(nl)
        nl = kl.Reshape((11, 11, 256), name="decoder_reshape")(nl)
        nl = kl.BatchNormalization(momentum=0.8, name="decoder_bn")(nl)
        nl = kl.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', name="decoder_dconv_1")(nl)
        nl = kl.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), output_padding=(1, 1), activation='relu', name="decoder_dconv_2")(nl)
        nl = kl.Reshape(target_shape, name='recon')(nl)
        return nl

    return decoder
