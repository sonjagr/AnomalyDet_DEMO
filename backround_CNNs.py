import tensorflow as tf
import os

br_1 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4,4), strides=(4,4), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(10, 10), strides=(10, 10),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(4, 4), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])
