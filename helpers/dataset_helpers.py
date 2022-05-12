import tensorflow as tf
import numpy as np
import pandas as pd
import math
from common import *

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
    #imgs = imgs.map(lambda x: tuple(tf.py_function(resize, [x], [tf.float32, ])))
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
