import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from capsnet import nn


def mask(inputs):
    """
    Mask data from all capsules except the most activated one, for each instance
    :param inputs: shape: (None, num_caps, dim_caps)
    :return:
    """
    norm = nn.norm(inputs, axis=-1)  # shape: (None, num_caps)
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
    norm = nn.norm(inputs, axis=-1)  # shape: (None, num_caps)
    # build index of elements to collect
    i = tf.range(start=0, limit=tf.shape(inputs)[0], delta=1)  # shape: (None, )
    j = tf.argmax(norm, axis=-1)  # shape: (None, )
    ij = tf.stack([i, tf.cast(j, tf.int32)], axis=1)
    # gather from index and return
    return tf.gather_nd(inputs, ij)


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
