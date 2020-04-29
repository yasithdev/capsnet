#!/usr/bin/env python3
import os
import sys

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras as k
from tensorflow.keras.datasets import mnist, cifar10, cifar100

from capsnet import losses
from models import get_model

# configuration
BASE_PATH = ""

# error messages
USAGE_EXPR = '''Usage:
    ./main  [ [MODE] train | retrain | test | demo ]
            [ [Dataset] mnist | cifar10 | cifar100 ]
            [ [MODEL NAME] original | deepcaps ]
'''
ERR_FILE_NOT_FOUND = "file not found"


def print_results(x_true, x_pred, y_true, y_pred, samples=20, cols=5):
    # define grid
    fig, axs = plt.subplots(ncols=cols, nrows=samples // cols + 1)
    # randomly select samples to plot
    for z in np.random.randint(0, len(x_true), samples):
        i = (z // cols) * 2
        j = z % cols
        axs[i, j].imshow(np.squeeze(x_pred), cmap='gray', vmin=0.0, vmax=1.0)
        axs[i, j].set_title(f'Label: {y_true[z]}, Predicted: {y_pred[z]}')
        axs[i, j].axis('off')
        axs[i + 1, j].imshow(np.squeeze(x_true), cmap='gray', vmin=0.0, vmax=1.0)
        axs[i + 1, j].axis('off')
    fig.show()


if __name__ == '__main__':
    # command-line arguments
    assert len(sys.argv) == 4, USAGE_EXPR
    mode = sys.argv[1].strip().lower()
    dataset_name = sys.argv[2].strip().lower()
    model_name = sys.argv[3].strip().lower()
    assert mode in ["train", "retrain", "test", "demo"], USAGE_EXPR
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
        x_train, x_test = x_train[..., None], x_test[..., None]
    if len(y_train.shape) > 1:
        y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
    # prepare for training
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = k.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = k.utils.to_categorical(y_test, NUM_CLASSES)

    # configure model and print summary
    model = get_model(name=model_name, input_shape=x_train.shape[1:], num_classes=NUM_CLASSES)
    model.compile(optimizer='adam',
                  loss=[losses.margin_loss, 'mse'],
                  loss_weights=[1, 5e-3],
                  metrics={'pred': 'acc'})
    model.summary(line_length=150)

    filepath = f"{BASE_PATH}weights_{model_name}_{dataset_name}.hdf5"

    if mode == "retrain":
        assert os.path.exists(filepath), ERR_FILE_NOT_FOUND
        model.load_weights(filepath)

    if mode == "train" or mode == "retrain":
        checkpoint = k.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        model.fit(x_train, [y_train, x_train],
                  batch_size=50,
                  epochs=5,
                  validation_data=(x_test, (y_test, x_test)),
                  callbacks=[checkpoint])

    if mode == "test":
        assert os.path.exists(filepath), ERR_FILE_NOT_FOUND
        model.load_weights(filepath)
        model.evaluate(x_test, [y_test, x_test])

    if mode == "demo":
        assert os.path.exists(filepath), ERR_FILE_NOT_FOUND
        model.load_weights(filepath)
        [y_pred, x_pred] = model.predict(x_test)
        print_results(x_test, x_pred, y_test, y_pred, samples=20, cols=5)
