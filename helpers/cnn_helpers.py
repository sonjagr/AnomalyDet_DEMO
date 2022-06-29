import cv2
import numpy as np
import tensorflow as tf
from common import *

## calculate the output dimensions of a conv layer
def cnn_layer_dim(input_size, kernel_size, strides, padding):
    output_size_x = ((input_size[0] - kernel_size[0] + 2*padding[0]) / strides[0]) + 1
    output_size_y = ((input_size[1] - kernel_size[1] + 2*padding[1]) / strides[1]) + 1
    return output_size_x, output_size_y

##calculate dynamic binary crossentropy
def dyn_weighted_bincrossentropy(true, pred):
    # get the total number of inputs
    num_pred = tf.keras.backend.sum(tf.keras.backend.cast(pred < 0.5, true.dtype)) + tf.keras.backend.sum(true)
    # get weight of values in 'pos' category
    zero_weight = tf.keras.backend.sum(true) / num_pred + tf.keras.backend.epsilon()
    # get weight of values in 'false' category
    one_weight = tf.keras.backend.sum(tf.keras.backend.cast(pred < 0.5, true.dtype)) / num_pred + tf.keras.backend.epsilon()
    # calculate the weight vector
    weights = (1.0 - true) * zero_weight + true * one_weight

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bin_crossentropy =bce(true, pred)

    weighted_bin_crossentropy = weights * bin_crossentropy
    return tf.keras.backend.mean(weighted_bin_crossentropy)

def weighted_bincrossentropy(true, pred, weight_zero=1.0, weight_one=100.):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bin_crossentropy = bce(true, pred)

    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return tf.keras.backend.mean(weighted_bin_crossentropy)

import matplotlib.pyplot as plt
def plot_metrics(history, savename):
    colors = ['blue','red']
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig('saved_CNNs/%s/training.png' % savename, dpi = DPI)
    plt.show()

def plot_examples(ds):
    for x, y in ds:
        dim = tf.shape(x)[-1]
        x = tf.reshape(x, [BOXSIZE, BOXSIZE, dim])
        x = x.numpy()
        plt.imshow(x.astype('uint8'), vmin = 0, vmax = 200.)
        plt.title(str(y))
        plt.show()

@tf.function
def crop(img, lbl):
    img = tf.reshape(img, [-1, PICTURESIZE_Y+16, PICTURESIZE_X, 1])
    img = tf.keras.layers.Cropping2D(cropping=((0, 16), (0, 0)))(img)
    return img, lbl

@tf.function
def flip(image_label, seed):
    image, label = image_label
    INPUT_DIM = tf.shape(image)[-1]
    image = tf.reshape(image, [BOXSIZE, BOXSIZE, INPUT_DIM])
    if seed > 5:
        flipped = tf.image.flip_left_right(image)
    else:
        flipped = tf.image.flip_up_down(image)
    return tf.reshape(flipped, [BOXSIZE, BOXSIZE, INPUT_DIM]), label

@tf.function
def patch_images(img, lbl):
    INPUT_DIM = tf.shape(img)[-1]
    split_img = tf.image.extract_patches(images=img, sizes=[1, BOXSIZE, BOXSIZE, 1], strides=[1, BOXSIZE, BOXSIZE, 1], rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [PATCHES, BOXSIZE *BOXSIZE, INPUT_DIM])
    lbl = tf.reshape(lbl, [PATCHES])
    patch_ds = tf.data.Dataset.from_tensors((re, lbl))
    return patch_ds

## calculate dataset length
def ds_length(ds):
    ds = ds.as_numpy_iterator()
    ds = list(ds)
    dataset_len = len(ds)
    return dataset_len

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

## a wrapper for rgb2bayer
def tf_rgb2bayer(image):
  im_shape = image.shape
  [image] = tf.py_function(rgb2bayer, [image], [tf.uint8])
  image.set_shape(im_shape)
  return tf.reshape(image, [-1])

## convert bayer to rgb
def bayer2rgb(bayer):
    bayer = bayer.numpy()
    shape = np.shape(bayer)
    rgb = cv2.cvtColor(bayer.reshape(shape[1],shape[2],1).astype('uint8'), cv2.COLOR_BAYER_RG2RGB)
    return rgb

## a wrapper for bayer2rgb
def tf_bayer2rgb(bayer):
    rgb = tf.py_function(bayer2rgb, [bayer], [tf.float32])
    return rgb

## calculate loss, use weights only when training
def loss(model, x, y, training):
    y_ = model(x, training=training)
    if training:
        loss_is = weighted_bincrossentropy(true=y, pred=y_)
    if not training:
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        loss_is = bce(y, y_)
    return loss_is

# calculate the gradient
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

@tf.function
def encode(img, lbl, ae):
    INPUT_DIM = tf.shape(img)[-1]
    img = tf.reshape(img, [-1, PICTURESIZE_Y, PICTURESIZE_X, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.abs(tf.subtract(img, decoded_img))
    return aed_img, lbl

@tf.function
def bright_encode(img, lbl, ae, delta):
    INPUT_DIM = tf.shape(img)[-1]
    img = tf.cast(img, tf.float64)
    img = tf.math.multiply(img, delta)
    img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=EIGHTBITMAX)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, [-1, PICTURESIZE_Y, PICTURESIZE_X, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.abs(tf.subtract(img, decoded_img))
    return aed_img, lbl

@tf.function
def encode_rgb(img, lbl, ae):
    img = tf.reshape(img, [-1, PICTURESIZE_Y, PICTURESIZE_X, 1])
    img_rgb = tf_bayer2rgb(img)
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    decoded_img_rgb = tf_bayer2rgb(decoded_img)
    aed_img = tf.abs(tf.subtract(img_rgb, decoded_img_rgb))
    return aed_img, lbl

@tf.function
def bright_encode_rgb(img, lbl, ae, delta):
    img = tf.cast(img, tf.float64)
    img = tf.math.multiply(img, delta)
    img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=EIGHTBITMAX)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, [-1, PICTURESIZE_Y, PICTURESIZE_X, 1])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    img_rgb = tf_bayer2rgb(img)
    decoded_img_rgb = tf_bayer2rgb(decoded_img)
    aed_img = tf.abs(tf.subtract(img_rgb, decoded_img_rgb))
    return aed_img, lbl

@tf.function
def rotate(image_label, rots):
    image, label = image_label
    INPUT_DIM = tf.shape(image)[-1]
    image = tf.reshape(image, [BOXSIZE, BOXSIZE, INPUT_DIM])
    rot = tf.image.rot90(image, k=rots)
    return tf.reshape(rot, [BOXSIZE, BOXSIZE, INPUT_DIM]), label

@tf.function
def crop_resize(image_label, seed):
    image, label = image_label
    INPUT_DIM = tf.shape(image)[-1]
    image = tf.image.random_crop(value=image, size = (BOXSIZE-20, BOXSIZE-20, INPUT_DIM), seed =seed)
    image = tf.image.resize(image, [BOXSIZE, BOXSIZE, INPUT_DIM])
    return image, label

@tf.function
def format_data(image, label):
    INPUT_DIM = tf.shape(image)[-1]
    image = tf.reshape(image, [BOXSIZE, BOXSIZE, INPUT_DIM])
    label = tf.cast(label, tf.float32)
    return image, label