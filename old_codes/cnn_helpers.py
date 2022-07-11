import cv2
import numpy as np
import random
from AE.autoencoders2 import *

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
    plt.savefig('saved_CNNs/%s/training.png' % savename, dpi = 600)
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

## flatten labels per whole image
@tf.function
def patch_labels(lbl):
    re = tf.reshape(lbl, [PATCHES])
    flat_ds = tf.data.Dataset.from_tensors((re))
    return flat_ds

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

## change brightness of patch
def bright_image_numpy(img):
    delta = random.choice([-0.2, 0.2, 0.01])
    img = tf.image.adjust_brightness(img, delta)
    return img

## a wrapper for bright_image
def tf_bright_image(image):
  im_shape = image.shape
  [image] = tf.py_function(bright_image_numpy, [image], [tf.float32])
  image.set_shape(im_shape)
  return image

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
def format_data(image, label):
    INPUT_DIM = tf.shape(image)[-1]
    image = tf.reshape(image, [BOXSIZE, BOXSIZE, INPUT_DIM])
    label = tf.cast(label, tf.float32)
    return image, label

'''
@tf.function
def process_crop_bright_encode(image_label, delta):
    image, label = image_label
    image, label = crop(image, label)
    image, label = bright_encode(image, label, ae, delta)
    return image,label

@tf.function
def process_crop_encode(image, label):
    image, label = crop(image, label)
    image, label = encode(image, label, ae)
    return image,label

@tf.function
def process_crop(image, label):
    image, label = crop(image, label)
    image = tf.reshape(image, [-1, 2720, 3840, INPUT_DIM])
    return image,label


## preprocessinh function for dataframes
def preprocess(imgs, lbls, to_encode = True, ae = False, normal_times = 50, to_augment = False, to_brightness = False, to_brightness_before_ae = False, batch_size = 256, drop_rem = False):

    cropped_imgs = imgs.map(crop)

    if to_brightness_before_ae:
        bright_cropped_imgs = cropped_imgs.map(tf_bayer2rgb)
        bright_dataset_changed = bright_cropped_imgs.map(bright_image_numpy)
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
'''




