# IMPORTS
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import imageio
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Conv2DTranspose
from keras.layers import concatenate

# HYPERPARAMERERS
EPOCHS = 5
VAL_SUBSPLITS = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32

# IMG SIZE
img_height = 96
img_width = 128
num_channels = 3

# DATA PATH
path = ''
image_path = os.path.join(path, './data/CameraRGB/')
mask_path = os.path.join(path, './data/CameraMask/')
image_list_orig = os.listdir(image_path)
image_list = [image_path + i for i in image_list_orig]
mask_list = [mask_path + i for i in image_list_orig]

# Split dataset into unmasked and masked images
image_list_ds = tf.data.Dataset.list_files(image_list, shuffle=False)
mask_list_ds = tf.data.Dataset.list_files(mask_list, shuffle=False)

image_filenames = tf.constant(image_list)
masks_filenames = tf.constant(mask_list)

dataset = tf.data.Dataset.from_tensor_slices(
    (image_filenames, masks_filenames))

# Preprocess the data


def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    # Normalizing image value between 0 and 1
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask


def preprocess(image, mask):  # Resize given image to shape (96, 128)
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')

    return input_image, input_mask


image_ds = dataset.map(process_path)
processed_image_ds = image_ds.map(preprocess)


def downsampling_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    # Encoder process - stacked of various conv_block
    # Each conv_block has 2 Conv2D layers with ReLU activation
    # With some specified blocks, we will apply Dropout or max_pooling method
    conv = Conv2D(n_filters,
                  3,
                  activation="relu",
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = MaxPooling2D((2, 2))(conv)
    else:
        next_layer = conv
    skip_connection = conv

    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    # Decoder process - unsamples the features back to original size
    # In each step, take the output and concatenate it before taking it to the next decoder block
    # expansive input is the input tensor from the previous layer
    # contractive input is the input tensor from previous skip layer

    up = Conv2DTranspose(
        n_filters,
        3,
        strides=(2, 2),
        padding='same')(expansive_input)

    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    return conv


def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):

    inputs = Input(input_size)
    cblock1 = downsampling_block(inputs, n_filters)
    cblock2 = downsampling_block(cblock1[0], 2 * n_filters)
    cblock3 = downsampling_block(cblock2[0], 4 * n_filters)

    cblock4 = downsampling_block(cblock3[0], 8 * n_filters, dropout_prob=0.3)
    cblock5 = downsampling_block(cblock4[0], 16 * n_filters,
                                 dropout_prob=0.3, max_pooling=False)

    ublock6 = upsampling_block(cblock5[0], cblock4[1],  n_filters * 8)

    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


unet = unet_model((img_height, img_width, num_channels))
unet.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(
                 from_logits=True),
             metrics=['accuracy'])

train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
model_history = unet.fit(train_dataset, epochs=EPOCHS)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_predictions(dataset=None, num=1):
    for image, mask in dataset.take(num):
        pred_mask = unet.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])


show_predictions(train_dataset, 6)
