import tensorflow as tf
import cv2
from common import *
import numpy as np
import random
import matplotlib.pyplot as plt

def cnn_layer_dim(input_size, kernel_size, strides, padding):
    output_size_x = ((input_size[0] - kernel_size[0] + 2*padding[0]) / strides[0]) + 1
    output_size_y = ((input_size[1] - kernel_size[1] + 2*padding[1]) / strides[1]) + 1
    return output_size_x, output_size_y

def dyn_weighted_bincrossentropy(true, pred):
    # get the total number of inputs
    num_pred = tf.keras.backend.sum(tf.keras.backend.cast(pred < 0.5, true.dtype)) + tf.keras.backend.sum(true)
    # get weight of values in 'pos' category
    zero_weight = tf.keras.backend.sum(true) / num_pred + tf.keras.backend.epsilon()
    # get weight of values in 'false' category
    one_weight = tf.keras.backend.sum(tf.keras.backend.cast(pred < 0.5, true.dtype)) / num_pred + tf.keras.backend.epsilon()
    # calculate the weight vector
    weights = (1.0 - true) * zero_weight + true * one_weight
    # calculate the binary cross entropy
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bin_crossentropy =bce(true, pred)
    # apply the weights
    weighted_bin_crossentropy = weights * bin_crossentropy
    return tf.keras.backend.mean(weighted_bin_crossentropy)

## crop bottom part
@tf.function
def crop(img):
    output = tf.image.crop_to_bounding_box(image = img, offset_height = 0, offset_width = 0, target_height = 2720, target_width= 3840)
    return output

# split into patches
@tf.function
def patch_images(image):
    split_img = tf.image.extract_patches(images=image, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1], rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [17*24, 160 * 160])
    noisy_ds = tf.data.Dataset.from_tensors((re))
    return noisy_ds

## flatten labels per whole image
@tf.function
def patch_labels(lbl):
    re = tf.reshape(lbl, [17*24])
    flat_ds = tf.data.Dataset.from_tensors((re))
    return flat_ds

## calculate dataset length
def ds_length(ds):
    ds = ds.as_numpy_iterator()
    ds = list(ds)
    dataset_len = len(ds)
    return dataset_len

## rotate anomalous patches
@tf.function
def rotate_image(x):
    x = tf.reshape(x, [1, 160, 160])
    rot_angle = random.choice([0, 1, 2])
    rot = tf.image.rot90(x, k=rot_angle, name=None)
    return tf.reshape(rot, [-1])

## convert rgb to bayer format
def rgb2bayer(rgb):
    rgb= rgb.numpy()
    (h,w) = rgb.shape[0], rgb.shape[1]
    (r,g,b) = cv2.split(rgb)
    bayer = np.empty((h, w), np.uint8)
    bayer[0::2, 0::2] = r[0::2, 0::2]
    bayer[0::2, 1::2] = g[0::2, 1::2]
    bayer[1::2, 0::2] = g[1::2, 0::2]
    bayer[1::2, 1::2] = b[1::2, 1::2]
    return bayer

def tf_rgb2bayer(image):
  im_shape = image.shape
  [image] = tf.py_function(rgb2bayer, [image], [tf.uint8])
  image.set_shape(im_shape)
  return tf.reshape(image, [-1])

## convert bayer to rgb
def bayer2rgb(bayer):
    rgb = cv2.cvtColor(bayer.numpy().reshape(160,160).astype('uint8'), cv2.COLOR_BAYER_RG2RGB)
    return rgb

def tf_bayer2rgb(bayer):
    rgb = tf.py_function(bayer2rgb, [bayer], [tf.float32])
    return rgb

## change brightness of patch
def bright_image(img):
    delta = random.choice([-0.2, 0.2, 0.01])
    img = tf.image.adjust_brightness(img, delta)
    return img

def tf_bright_image(image):
  im_shape = image.shape
  [image] = tf.py_function(bright_image, [image], [tf.float32])
  image.set_shape(im_shape)
  return image

def loss(model, x, y, training):
    y_ = model(x, training=training)
    if training:
        loss_is = dyn_weighted_bincrossentropy(true=y, pred=y_)
    if not training:
        loss_is = tf.keras.losses.BinaryCrossentropy(y, y_, from_logits=False)
    return loss_is

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)







