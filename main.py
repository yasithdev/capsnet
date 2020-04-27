#!/usr/bin/env python3
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.datasets import mnist, cifar10, cifar100

from capsnet import losses, metrics
from functions import print_results
from models import get_model

# configuration
BASE_PATH = ""

# error messages
USAGE_EXPR = '''Usage:
    ./main  [ [MODE] train | test | demo ]
            [ [Dataset] mnist | cifar10 | cifar100 ]
            [ [MODEL NAME] original | deepcaps ]
'''
ERR_FILE_NOT_FOUND = "file not found"

if __name__ == '__main__':
    # command-line arguments
    assert len(sys.argv) == 4, USAGE_EXPR
    mode = sys.argv[1].strip().lower()
    dataset_name = sys.argv[2].strip().lower()
    model_name = sys.argv[3].strip().lower()
    assert mode in ["train", "test", "demo"], USAGE_EXPR
    assert dataset_name in ["mnist", "cifar10", "cifar100"], USAGE_EXPR
    assert model_name in ["original", "deepcaps"], USAGE_EXPR

    # set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # load data
    if dataset_name == "mnist": dataset = mnist
    if dataset_name == "cifar10": dataset = cifar10
    if dataset_name == "cifar100": dataset = cifar100

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    NUM_CLASSES = len(np.unique(y_train))

    # transform data for training
    if len(x_train.shape) == 3:
        x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]
    # prepare for training
    x_train = tf.divide(x_train, 255.0)
    x_test = tf.divide(x_test, 255.0)
    y_train = tf.one_hot(y_train, NUM_CLASSES)
    y_test = tf.one_hot(y_train, NUM_CLASSES)

    # configure model and print summary
    model = get_model(input_shape=x_train.shape[1:], name=model_name)
    model.compile(optimizer='adam', loss=[losses.margin_loss, losses.reconstruction_loss], loss_weights=[1, 5e-3],
                  metrics={'margin': metrics.accuracy})
    model.summary(line_length=150)

    filepath = f"{BASE_PATH}{model_name}_weights.hdf5"

    if mode == "train":
        checkpoint = k.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        model.fit(x_train, [y_train, x_train], batch_size=50, epochs=5, validation_split=0.1, callbacks=[checkpoint])

    if mode == "test":
        assert os.path.exists(filepath), ERR_FILE_NOT_FOUND
        model.load_weights(filepath)
        model.evaluate(x_test, [y_test, x_test])

    if mode == "demo":
        assert os.path.exists(filepath), ERR_FILE_NOT_FOUND
        model.load_weights(filepath)
        [y_pred, x_pred] = model.predict(x_test)
        print_results(x_test, x_pred, y_test, y_pred, samples=20, cols=5)
