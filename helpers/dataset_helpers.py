import tensorflow as tf
import numpy as np
import pandas as pd
import math
from common import *
import random
import cv2

def process_anomalous_df_to_numpy(db):
    db = db[db['orig_boxY'].map(len) > 0]
    db = db.reset_index()[['Campaign', 'DUT', 'FileName', 'orig_boxY', 'orig_boxX']]
    db = db.drop(db.loc[(db.Campaign == 'September2021_PM8') & (db.DUT == '8inch_198ch_N3311_7')].index)
    db['Path'] = db.Campaign + '/' + db.DUT + '/' + db.FileName
    db = db.drop(['Campaign', 'DUT', 'FileName'], axis =1)
    db['crop_lbls'] = pd.NaT
    db['crop_lbls'] = db.apply(lambda x: box_to_labels(x.orig_boxX, x.orig_boxY).flatten(), axis=1)
    X_list = db['Path'].to_numpy()
    Y_list = db['crop_lbls'].to_numpy().flatten()
    return X_list, Y_list

## resize images to the required size, crop from bottom
def resize(item):
    model = tf.keras.Sequential([tf.keras.layers.Cropping2D(cropping=((0, 16),(0, 0)))])
    return model(item)

## read npy file for dataset
def read_npy_file(item):
    data = np.load(item.numpy().decode())
    data = np.expand_dims(data, axis=2)
    return data.astype(np.float32)

def tf_read_npy_file(item):
  im_shape = item.shape
  [item] = tf.py_function(read_npy_file, [item], [tf.float32])
  item.set_shape(im_shape)
  return [item]

## create a general dataset
def create_dataset(file_list, _shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    if _shuffle:
        dataset.shuffle(len(file_list), reshuffle_each_iteration=True)
    return dataset.map(lambda item: tuple(tf.py_function(read_npy_file, [item], [tf.float32,])))

## creates a training dataset for the CNN
def create_cnn_dataset(file_list, label_list, _shuffle=False):
    imgs = tf.data.Dataset.from_tensor_slices(file_list)
    lbls = tf.data.Dataset.from_tensor_slices(label_list)
    dataset = tf.data.Dataset.zip((imgs, lbls))
    if _shuffle:
        dataset.shuffle(len(file_list), seed = 42, reshuffle_each_iteration=True)
    images = dataset.map(lambda x, y: x)
    labels = dataset.map(lambda x, y: y)
    images = images.map(lambda item: tuple(tf.py_function(read_npy_file, [item], [tf.float32,])))
    dataset = tf.data.Dataset.zip((images, labels))
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





