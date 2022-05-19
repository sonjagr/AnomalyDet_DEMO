import tensorflow as tf
import cv2
from common import *
import numpy as np
import random
import matplotlib.pyplot as plt

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
    rot_angle = random.choice([1, 2, 3])
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

## a wrapper for rgb2bayer
def tf_rgb2bayer(image):
  im_shape = image.shape
  [image] = tf.py_function(rgb2bayer, [image], [tf.uint8])
  image.set_shape(im_shape)
  return tf.reshape(image, [-1])

## convert bayer to rgb
def bayer2rgb(bayer):
    rgb = cv2.cvtColor(bayer.numpy().reshape(160,160).astype('uint8'), cv2.COLOR_BAYER_RG2RGB)
    return rgb

## a wrapper for bayer2rgb
def tf_bayer2rgb(bayer):
    rgb = tf.py_function(bayer2rgb, [bayer], [tf.float32])
    return rgb

## change brightness of patch
def bright_image(img):
    delta = random.choice([-0.2, 0.2, 0.01])
    img = tf.image.adjust_brightness(img, delta)
    return img

## a wrapper for bright_image
def tf_bright_image(image):
  im_shape = image.shape
  [image] = tf.py_function(bright_image, [image], [tf.float32])
  image.set_shape(im_shape)
  return image

## calculate loss, use weights only when training
def loss(model, x, y, training):
    y_ = model(x, training=training)
    if training:
        loss_is = dyn_weighted_bincrossentropy(true=y, pred=y_)
    if not training:
        loss_is = tf.keras.losses.BinaryCrossentropy(y, y_, from_logits=False)
    return loss_is

# calculate the gradient
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

## autoencode and return difference
@tf.function
def encode(img, ae):
    img = tf.reshape(img, [-1, 2720, 3840, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.sqrt(tf.pow(tf.subtract(img, decoded_img), 2))
    return aed_img

## preprocessinh function for dataframes
def preprocess(imgs, lbls, to_encode = True, ae = False, normal_times = 50, to_augment = False, to_brightness = False, to_brightness_before_ae = False, batch_size = 256, drop_rem = False):

    cropped_imgs = imgs.map(crop)

    if to_brightness_before_ae:
        bright_cropped_imgs = cropped_imgs.map(tf_bayer2rgb)
        bright_dataset_changed = bright_cropped_imgs.map(bright_image)
        bright_cropped_images = bright_dataset_changed.map(tf_rgb2bayer)

    if to_encode == True:
        diff_imgs = cropped_imgs.map(lambda x: encode(x, ae))
        if to_brightness_before_ae == True:
            bright_diff_imgs = bright_cropped_imgs.map(lambda x: encode(x, ae))
    if to_encode == False:
        diff_imgs = cropped_imgs

    patched_imgs = diff_imgs.flat_map(patch_images)
    patched_lbls = lbls.flat_map(patch_labels)

    patched_imgs = patched_imgs.unbatch()
    patched_lbls = patched_lbls.unbatch()

    if to_brightness_before_ae == True:
        bright_patched_imgs = bright_diff_imgs.flat_map(patch_images)
        bright_patched_imgs = bright_patched_imgs.unbatch()

    dataset = tf.data.Dataset.zip((patched_imgs, patched_lbls))

    anomalous_dataset = dataset.filter(lambda x, y: y == 1.)
    anomalous_nbr_before = ds_length(anomalous_dataset)
    anomalous_images = anomalous_dataset.map(lambda x, y: x)
    anomalous_labels = anomalous_dataset.map(lambda x, y: y)

    if to_brightness_before_ae == True:
        bright_dataset = tf.data.Dataset.zip((bright_patched_imgs, patched_lbls))
        bright_anomalous_dataset = bright_dataset.filter(lambda x, y: y == 1.)
        bright_anomalous_images = bright_anomalous_dataset.map(lambda x, y: x)
        bright_anomalous_labels = bright_anomalous_dataset.map(lambda x, y: y)

    if to_augment == True:
        rotated_dataset = anomalous_images.map(lambda x: rotate_image(x))
        rotated_anomalous_dataset = tf.data.Dataset.zip((rotated_dataset, anomalous_labels))
        combined_anomalous_dataset = anomalous_dataset.concatenate(rotated_anomalous_dataset)
        if to_brightness == True:
            bright_dataset_rgb = anomalous_images.map(tf_bayer2rgb)
            bright_dataset_changed = bright_dataset_rgb.map(bright_image)
            bright_dataset_bayer = bright_dataset_changed.map(tf_rgb2bayer)
            bright_anomalous_dataset = tf.data.Dataset.zip((bright_dataset_bayer, anomalous_labels))
            bright_anomalous_dataset = bright_anomalous_dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
            combined_anomalous_dataset = combined_anomalous_dataset.concatenate(bright_anomalous_dataset)
    if to_augment == False:
        combined_anomalous_dataset = anomalous_dataset
    anomalous_nbr = ds_length(combined_anomalous_dataset)
    normal_dataset = dataset.filter(lambda x, y: y == 0.).shuffle(500).take(normal_times * anomalous_nbr)

    combined_dataset_batch = normal_dataset.concatenate(combined_anomalous_dataset).shuffle(buffer_size = 20000, reshuffle_each_iteration = True).batch(batch_size=batch_size, drop_remainder=drop_rem)
    return combined_dataset_batch, anomalous_nbr_before, anomalous_nbr





