import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle


def compose_image(digit, background):
    """Difference-blend a digit and a random patch from a background image."""
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)
    bg = background[x:x+dw, y:y+dh]
    return np.abs(bg - digit).astype(np.uint8)

def mnist_to_img(x):
    """Binarize MNIST digit and convert to RGB."""
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)

def create_mnistm(X):
    """
    Give an array of MNIST digits, blend random background patches to
    build the MNIST-M dataset.
    """
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):
        bg_img = np.random.choice(background_data)
        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img)
        X_[i] = d
    return X_

def imshow_grid(images, shape=[2, 8]):
    fig, axes = plt.subplots(shape[0], shape[1], figsize=(12, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')
    plt.show()

def difference_loss(private_samples, shared_samples, weight=0.05, name=''):
    private_samples -= tf.reduce_mean(private_samples, 0)
    shared_samples -= tf.reduce_mean(shared_samples, 0)
    private_samples = tf.keras.backend.l2_normalize(private_samples, 1)
    shared_samples = tf.keras.backend.l2_normalize(shared_samples, 1)
    correlation_matrix = tf.matmul(private_samples, shared_samples, transpose_a=True)
    cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
    cost = tf.where(cost > 0, cost, 0, name='value')
    assert_op = tf.Assert(tf.is_finite(cost), [cost])
    with tf.control_dependencies([assert_op]):
        tf.losses.add_loss(cost)
    return cost

def concat_operation(shared_repr, private_repr):
    return shared_repr + private_repr

@tf.custom_gradient
def flip_gradient(x, l=1.0):
    y = tf.identity(x)
    def grad(dy):
        return tf.negative(dy) * l, None
    return y, grad

def shuffle_aligned_list(data):
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


import numpy as np
def batch_generator(data, batch_size, shuffle=True):
    """
    Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    
    Args:
        data: A list of array-like objects (e.g., features, labels)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data before creating batches
        
    Yields:
        A list of array-like objects corresponding to a batch of data
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        start = batch_count * batch_size
        end = start + batch_size
        if end > len(data[0]):  # Check if end index exceeds data size
            break  # Exit the loop to avoid smaller batches
        batch_count += 1
        yield [d[start:end] for d in data]


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def imshow_grid(images, cmap=None,shape=[2, 8],title=''):
    fig, axes = plt.subplots(shape[0], shape[1], figsize=(12, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(title, fontsize=16)
    plt.show()
