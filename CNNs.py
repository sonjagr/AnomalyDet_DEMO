import tensorflow as tf
from common import *

model_works_newdatasplit = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=8, kernel_size=(4,4), strides=(4,4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu',),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_works_simplest = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=8, kernel_size=(4,4), strides=(4,4), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4) , padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=(2,2), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu',),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_small = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(10,10), strides=(8,8)  , padding="valid", activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(4,4) , padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(2,2), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_init = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4,4), strides=(2,2), padding="valid",activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(4,4) , padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(4, 4), padding="valid", activation='relu',),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_init2 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4,4), strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2),  kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(8,8), strides=(4,4) ,  kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(8, 8),  kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid", activation='relu',),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_init2_lessr = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4,4), strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2),  kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(8,8), strides=(4,4),  padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(8, 8), padding="valid", activation='relu',),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_init2_lessr2 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4,4), strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2),  kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01),  padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(8, 8), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid", activation='relu',),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])


model_init2_lessr2_larger = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=8, kernel_size=(4,4), strides=(4,4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    #tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01),  padding="valid", activation='relu'),
    #tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(5, 5),  padding="valid", activation='relu',),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_init2_lessr2_larger_noae = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4,4), strides=(4,4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(5, 5),  padding="valid", activation='relu',),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_works3 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=8, kernel_size=(4,4), strides=(4,4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(5, 5),  padding="valid", activation='relu',),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])


model_init2_lessr2_larger2 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=8, kernel_size=(4,4), strides=(4,4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(5, 5), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid", activation='relu',),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_init2_lessr2_larger3 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4,4), strides=(4,4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4), kernel_regularizer=tf.keras.regularizers.l2(l=0.01),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(5, 5), kernel_regularizer=tf.keras.regularizers.l2(l=0.01), padding="valid", activation='relu',),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])


'''
model_init3 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=8, kernel_size=(2,2), strides=(2,2), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4,4), strides=(2,2), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2) , padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(19,19), strides=(19,19), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])
'''

model_init3 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(10,10), strides=(10,10), padding="valid",activation='selu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='selu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), padding="valid", activation='selu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(4,4), padding="valid", activation = 'selu',),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

