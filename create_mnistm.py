from tensorflow.keras.datasets import mnist
import tarfile
import os
import hickle as hkl
import numpy as np
import pandas as pd
import skimage
import skimage.io
import skimage.transform

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

BST_PATH = '/content/drive/MyDrive/BSR_bsds500.tgz'

rand = np.random.RandomState(42)

f = tarfile.open(BST_PATH)
train_files = []
for name in f.getnames():
    if name.startswith('BSR/BSDS500/data/images/train/'):
        train_files.append(name)

print('Loading BSR training images')
background_data = []
for name in train_files:
    try:
        fp = f.extractfile(name)
        bg_img = skimage.io.imread(fp)
        background_data.append(bg_img)
    except:
        continue


def compose_image(digit, background):
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)
    bg = background[x:x + dw, y:y + dh]
    return np.abs(bg - digit).astype(np.uint8)


def mnist_to_img(x):
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)


def create_mnistm(X):
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):
        if i % 1000 == 0:
            print(i)
        bg_index = rand.choice(len(background_data))  # Randomly select an index
        bg_img = background_data[bg_index]           # Access the background image using the index

        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img)
        X_[i] = d
    return X_

def savefile(history,path):
#    if not os.path.exists(path):
#        os.makedirs(path)
    hkl.dump(history,path)
path='/content/drive/MyDrive'    
print('Building train set...')
train = create_mnistm(train_images)
print('Building test set...')
test = create_mnistm(test_images)
print('Building validation set...')
valid = create_mnistm(test_images[:5000])  # Assuming you want to use a subset of test images for validation

mnist_data={'train':train,'test':test,'valid':valid}
# Save dataset as hickle file
path = os.path.join(str(path), 'mnistm_data.hkl')
savefile(mnist_data, path)
