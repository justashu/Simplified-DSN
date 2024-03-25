import tensorflow as tf
from tensorflow.keras import layers

def shared_encoder(x, name='shared_encoder'):
    with tf.name_scope(name) as scope:
        # Define layers
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1_shared_encoder')
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1_shared_encoder')
        conv2 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', name='conv2_shared_encoder')
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2_shared_encoder')
        flatten = tf.keras.layers.Flatten(name='flat_shared_encoder')
        fc1 = tf.keras.layers.Dense(100, activation='relu', name='shared_fc1')
        
        # Apply layers
        net = conv1(x)
        net = pool1(net)
        net = conv2(net)
        net = pool2(net)
        net = flatten(net)
        net = fc1(net)

     # Set layers trainable
        conv1.trainable = True
        conv2.trainable = True
        fc1.trainable = True
        
    return net

def private_target_encoder(x, name='private_target_encoder'):
    with tf.name_scope(name) as scope:
        # Define layers
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1_private_target_encoder')
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1_private_target_encoder')
        conv2 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', name='conv2_private_target_encoder')
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool2_private_target_encoder')
        flatten = tf.keras.layers.Flatten(name='flat_private_target_encoder')
        fc1 = tf.keras.layers.Dense(100, activation='relu', name='private_target_fc1')
        
        # Apply layers
        net = conv1(x)
        net = pool1(net)
        net = conv2(net)
        net = pool2(net)
        net = flatten(net)
        net = fc1(net)
        
         # Set layers trainable
        conv1.trainable = True
        conv2.trainable = True
        fc1.trainable = True
    return net


def private_source_encoder(x, name='private_source_encoder'):
    with tf.name_scope(name) as scope:
        # Define layers
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1_private_source_encoder')
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1_private_source_encoder')
        conv2 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', name='conv2_private_source_encoder')
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2_private_source_encoder')
        flatten = tf.keras.layers.Flatten(name='flat_private_source_encoder')
        fc1 = tf.keras.layers.Dense(100, activation='relu', name='private_source_fc1')
        
        # Apply layers
        net = conv1(x)
        net = pool1(net)
        net = conv2(net)
        net = pool2(net)
        net = flatten(net)
        net = fc1(net)

     # Set layers trainable
        conv1.trainable = True
        conv2.trainable = True
        fc1.trainable = True
    return net

def shared_decoder(feat, height, width, channels, name='shared_decoder'):
    with tf.name_scope(name) as scope:
        # Define layers
        fc1 = tf.keras.layers.Dense(600, activation='relu', name='fc1_decoder')
        reshape = tf.keras.layers.Reshape((10, 10, 6))
        conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1_1_decoder')
        resize1 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (16, 16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        conv2 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv2_1_decoder')
        resize2 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (32, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        conv3 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv3_2_decoder')
        resize3 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        conv4 = tf.keras.layers.Conv2D(channels, kernel_size=(3, 3), activation=None, padding='same', name='conv4_1_decoder')
        
        # Apply layers
        net = fc1(feat)
        net = reshape(net)
        net = conv1(net)
        net = resize1(net)
        net = conv2(net)
        net = resize2(net)
        net = conv3(net)
        net = resize3(net)
        net = conv4(net)

     # Set layers trainable
        fc1.trainable = True
        conv1.trainable = True
        conv2.trainable = True
        conv3.trainable = True
        conv4.trainable = True
    return net




# def shared_decoder(feat, height, width, channels, name='shared_decoder'):
#     with tf.name_scope(name) as scope:
#         # Define layers
#         fc1 = tf.keras.layers.Dense(600, activation='relu', name='fc1_decoder')
#         reshape = tf.keras.layers.Reshape((10, 10, 6))
#         conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv1_1_decoder')
#         resize1 = tf.keras.layers.UpSampling2D(size=(2, 2))
#         conv2 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv2_1_decoder')
#         resize2 = tf.keras.layers.UpSampling2D(size=(2, 2))
#         conv3 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', name='conv3_2_decoder')
#         resize3 = tf.keras.layers.UpSampling2D(size=(2, 2))
#         conv4 = tf.keras.layers.Conv2D(channels, kernel_size=(3, 3), activation=None, padding='same', name='conv4_1_decoder')
        
#         # Apply layers
#         net = fc1(feat)
#         net = reshape(net)
#         net = conv1(net)
#         net = resize1(net)
#         net = conv2(net)
#         net = resize2(net)
#         net = conv3(net)
#         net = resize3(net)
#         net = conv4(net)
#     return net
