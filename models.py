import tensorflow as tf
from tensorflow.keras import layers

# Shared Encoder
def shared_encoder(x, name='feat_ext', reuse=False):
    with tf.name_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        with tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1_shared_encoder'):
            net = layers.MaxPooling2D(pool_size=(2, 2), name='pool1_shared_encoder')(net)
            net = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', name='conv2_shared_encoder')(net)
            net = layers.MaxPooling2D(pool_size=(2, 2), name='pool2_shared_encoder')(net)
            net = layers.Flatten(name='flat_shared_encoder')(net)
            net = layers.Dense(100, activation='relu', name='shared_fc1')(net)
    return net

# Private Target Encoder
def private_target_encoder(x, name='priviate_target_encoder', reuse=False):
    with tf.name_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        with tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1'):
            net = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1')(net)
            net = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', name='conv2')(net)
            net = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool2')(net)
            net = layers.Flatten()(net)
            net = layers.Dense(100, activation='relu', name='private_target_fc1')(net)
    return net

# Private Source Encoder
def private_source_encoder(x, name='priviate_source_encoder', reuse=False):
    with tf.name_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        with tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1'):
            net = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(net)
            net = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', name='conv2')(net)
            net = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(net)
            net = layers.Flatten()(net)
            net = layers.Dense(100, activation='relu', name='private_source_fc1')(net)
    return net

# Shared Decoder
def shared_decoder(feat, height, width, channels, reuse=False, name='shared_decoder'):
    with tf.name_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        with tf.keras.layers.Dense(600, activation='relu', name='fc1_decoder'):
            net = tf.reshape(feat, [-1, 10, 10, 6])
            net = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1_1_decoder')(net)
            net = tf.image.resize(net, (16, 16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            net = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv2_1_decoder')(net)
            net = tf.image.resize(net, (32, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            net = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv3_2_decoder')(net)
            output_size = [height, width]
            net = tf.image.resize(net, output_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            net = tf.keras.layers.Conv2D(channels, kernel_size=(3, 3), activation=None, padding='same', name='conv4_1_decoder')(net)
    return net
