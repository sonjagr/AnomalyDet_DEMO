import tensorflow as tf
import numpy as np
import pandas as pd
import math
from common import *
from scipy.ndimage.interpolation import rotate
import random
import cv2

## resize images to the required size, crop from bottom
def resize(item):
    model = tf.keras.Sequential([tf.keras.layers.Cropping2D(cropping=((0, 16),(0, 0)))])
    return model(item)

## read npy file for dataset
def read_npy_file(item):
    data = np.load(item.numpy().decode())
    data = np.expand_dims(data, axis=2)
    return data.astype(np.float32)

## create a general dataset
def create_dataset(file_list, _shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    if _shuffle:
        dataset.shuffle(len(file_list), reshuffle_each_iteration=True)
    return dataset.map(lambda item: tuple(tf.py_function(read_npy_file, [item], [tf.float32,])))

## creates a training dataset for the CNN
def create_cnn_dataset(file_list, label_list, _shuffle=False):
    imgs = tf.data.Dataset.from_tensor_slices(file_list)
    imgs = imgs.map(lambda x: tuple(tf.py_function(read_npy_file, [x], [tf.float32, ])))
    labels = tf.data.Dataset.from_tensor_slices(label_list)
    dataset = tf.data.Dataset.zip((imgs, labels))
    if _shuffle:
        dataset.shuffle(len(file_list), reshuffle_each_iteration=True)
    return dataset

## convert box coordinates to labels
def box_to_labels(BoxX, BoxY):
    labels = np.zeros((YS, XS))
    for i, x in enumerate(BoxX):
        x = int(x)
        y = int(BoxY[i])
        if (np.mod(x, 160) == 0) and (np.mod(y, 160) == 0):
            xi = int(x / 160)
            yi = int(y / 160)
            labels[yi, xi] = 1
        else:
            print('Error in annotation conversion')
    return labels

## convert box index to the coordinates
def box_index_to_coords(box_index):
    row = math.floor(box_index / 24)
    col = box_index % 24
    return col*160, row*160

def combine_dbs(files, openloc, saveloc, name):
    files = ['EndOf2021_PM8.h5', 'Fall2021_PM8.h5', 'LongTermIV_2021_ALPS.h5', 'September2021_PM8.h5',
             'Winter2022_ALPS.h5']
    openloc = DataBaseFileLocation_local
    name = 'main_db_bb_crop'
    saveloc = openloc + name + '.h5'
    append_db = []
    for f in files:
        with pd.HDFStore(openloc, mode='r') as store:
            db = store.select('db')
        append_db.append(db)
        print(f'Reading {f}')
    main_db_bb_crop = pd.concat(append_db)
    main_db_bb_crop.to_hdf(saveloc, key=name, mode='w')

## function to add rotation to images
def rotate_img(img):
    img = img.reshape(160,160)
    rot_angle = random.choice([90, 180, 270])
    rot = rotate(img, rot_angle)
    return rot.flatten()

## convert rgb to bayer format
def rgb2bayer(rgb):
    (h,w) = rgb.shape[0], rgb.shape[1]
    (r,g,b) = cv2.split(rgb)
    bayer = np.empty((h,w), np.uint8)
    bayer[0::2, 0::2] = r[0::2, 0::2]
    bayer[0::2, 1::2] = g[0::2, 1::2]
    bayer[1::2, 0::2] = g[1::2, 0::2]
    bayer[1::2, 1::2] = b[1::2, 1::2]
    return bayer

## convert bayer to rgb
def bayer2rgb(bayer):
    return cv2.cvtColor(bayer.astype('uint8'), cv2.COLOR_BAYER_RG2RGB)

## change the brightness of whole images
def change_brightness(img, value):
    value = value.astype('uint8')
    rgb = bayer2rgb(img.reshape(2736, 3840))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    final_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    final_bayer = rgb2bayer(final_rgb)
    return final_bayer.reshape(-1, 2736, 3840, 1)

## change the brightness of patches
def change_brightness_patch(img):
    value = random.choice(np.arange(-51,51,1)).astype('uint8')
    rgb = bayer2rgb(img.reshape(160, 160))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    final_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    final_bayer = rgb2bayer(final_rgb)
    return final_bayer.reshape(160, 160).flatten()

## function to calculate autoencoder reconstruction error as absolute error
def encode(ae, img):
    img = img[0].numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img).numpy()
    aed_img = np.sqrt(np.power(np.subtract(img, decoded_img),2))
    return aed_img

## split image into patches
def split(img):
    img = img[0].numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
    img = tf.convert_to_tensor(img, np.float32)
    split_img = tf.image.extract_patches(images=img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1],rates=[1, 1, 1, 1], padding='VALID')
    return split_img.numpy().reshape(17 * 24, 160 * 160)
