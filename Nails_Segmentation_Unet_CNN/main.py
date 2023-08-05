import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Model
from keras.callbacks import ModelCheckpoint
import seaborn as sns

from skimage.io import imread, imshow
from skimage.transform import resize


# Constants
TRAIN_PATH = 'nails_segmentation/images/'
IMG_SIZE = 256


# Preprocessing X_train
train_ids = glob(TRAIN_PATH+'*.jpg')
X_train = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.float16)
for n, id_ in tqdm(enumerate(train_ids), total= len(train_ids)):
    img = imread(id_)[:,:,:3]
    img = resize(img, (IMG_SIZE, IMG_SIZE), mode= 'constant', preserve_range= True)
    X_train[n] = img

plt.figure(figsize=(20,20))
for n ,i in enumerate(list(np.random.randint(0,len(X_train),16))) :
    plt.subplot(4,4,n+1)
    plt.imshow(X_train[i])
    plt.axis('off')
    plt.title(i)


# Preprocessing Y_train
TRAIN_PATH = 'nails_segmentation/labels/'
train_ids = glob(TRAIN_PATH+'*.jpg')
Y_train = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, 1), dtype= np.bool)
for n, id_ in tqdm(enumerate(train_ids), total= len(train_ids)):
    mask = imread(id_)[:,:,:1]
    mask = resize(mask, (IMG_SIZE, IMG_SIZE), mode= 'constant', preserve_range= True)
    Y_train[n] = mask

plt.figure(figsize=(20, 20))
for n , i in enumerate(list(np.random.randint(0, len(Y_train), 16))):
    plt.subplot(4, 4, n+1)
    plt.imshow(Y_train[i], cmap='gray')
    plt.axis('off')
    plt.title(i)


# Building Model
model = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same',),
    layers.Dropout(0.1),
    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.2),
    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.2),
    layers.Conv2D(256, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding= 'same'),
    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.2),
    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Conv2DTranspose(64, (2, 2), strides= (2, 2), padding= 'same'),
    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.1),
    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Conv2D(1, (1, 1), activation='sigmoid')
])

model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])


# Callbacks
checkpoint = ModelCheckpoint('output/checkpoint.h5',
                            monitor= 'val_loss',
                            verbose= 1,
                            save_best_only= True,
                            save_weights_only= True)


# Model Training
history = model.fit(X_train, Y_train,
                    validation_split= 0.1,
                    batch_size= 10,
                    steps_per_epoch= len(X_train)//10,
                    epochs = 50,
                    callbacks= checkpoint,
                    verbose= 1,
                    shuffle= True)


# Accuracy Visualization
sns.set()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()