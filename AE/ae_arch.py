## plot AE architecture
import tensorflow as tf
import numpy as np

encoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(2720, 3840, 1)),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 4), strides=(5, 4), activation='elu', padding="valid", use_bias=True),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2), activation='elu', padding="valid", use_bias=True),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 2), strides=(4, 2), activation='elu', padding="valid", use_bias=True),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 5), strides=(2, 5), activation='elu', padding="valid", use_bias=True),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation='elu', padding="valid", use_bias=True),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), activation='elu', padding="valid", use_bias=True),])

decoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(17, 24, 16)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), activation='elu', padding="valid", use_bias=True),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), activation='elu', padding="valid", use_bias=True),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 5), strides=(2, 5), activation='elu', padding="valid", use_bias=True),
        tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(4, 2), strides=(4, 2), activation='elu', padding="valid", use_bias=True),
        tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding="valid", use_bias=True),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(5, 4), strides=(5, 4), activation='relu', padding="valid", use_bias=True),
        tf.keras.layers.experimental.preprocessing.Rescaling(255),])

print(encoder.summary())
print(decoder.summary())
