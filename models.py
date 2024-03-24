import tensorflow as tf
from tensorflow.keras import layers

# Shared Encoder
def shared_encoder(x, name='shared_encoder', reuse=False):
    with tf.name_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        # Define layers
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1_shared_encoder')
        pool1 = layers.MaxPooling2D(pool_size=(2, 2), name='pool1_shared_encoder')
        conv2 = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', name='conv2_shared_encoder')
        pool2 = layers.MaxPooling2D(pool_size=(2, 2), name='pool2_shared_encoder')
        flatten = layers.Flatten(name='flat_shared_encoder')
        fc1 = layers.Dense(100, activation='relu', name='shared_fc1')
        
        # Apply layers
        net = conv1(x)
        net = pool1(net)
        net = conv2(net)
        net = pool2(net)
        net = flatten(net)
        net = fc1(net)
    return net


# Private Target Encoder
def private_target_encoder(x, name='private_target_encoder', reuse=False):
    with tf.name_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        # Define layers
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1')
        pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1')
        conv2 = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', name='conv2')
        pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool2')
        flatten = layers.Flatten()
        fc1 = layers.Dense(100, activation='relu', name='private_target_fc1')
        
        # Apply layers
        net = conv1(x)
        net = pool1(net)
        net = conv2(net)
        net = pool2(net)
        net = flatten(net)
        net = fc1(net)
    return net


# Private Source Encoder
def private_source_encoder(x, name='private_source_encoder', reuse=False):
    with tf.name_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        # Define layers
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1')
        pool1 = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')
        conv2 = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', name='conv2')
        pool2 = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')
        flatten = layers.Flatten()
        fc1 = layers.Dense(100, activation='relu', name='private_source_fc1')
        
        # Apply layers
        net = conv1(x)
        net = pool1(net)
        net = conv2(net)
        net = pool2(net)
        net = flatten(net)
        net = fc1(net)
    return net


# Shared Decoder
def shared_decoder(feat, height, width, channels, reuse=False, name='shared_decoder'):
    with tf.name_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        # Define layers
        fc1 = tf.keras.layers.Dense(600, activation='relu', name='fc1_decoder')
        conv1_1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1_1_decoder')
        conv2_1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv2_1_decoder')
        conv3_2 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv3_2_decoder')
        conv4_1 = tf.keras.layers.Conv2D(channels, kernel_size=(3, 3), activation=None, padding='same', name='conv4_1_decoder')
        
        # Apply layers
        net = fc1(feat)
        net = tf.reshape(net, [-1, 10, 10, 6])
        net = conv1_1(net)
        net = tf.image.resize(net, (16, 16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        net = conv2_1(net)
        net = tf.image.resize(net, (32, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        net = conv3_2(net)
        output_size = [height, width]
        net = tf.image.resize(net, output_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        net = conv4_1(net)
    return net

