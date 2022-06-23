import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

model_whole = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), padding="valid",activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(4, 4),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_whole_final3 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), strides=(2,2), padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2),  padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),
])


model_whole_final2 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),

    tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=(2,2), padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2),  padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),
])

model_whole_final = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),

    tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=(2,2), padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(4, 4),  padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),
])


model_whole_larger = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(4,4), padding="valid",activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=512, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_whole_smaller = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=(2,2), padding="valid",activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_whole_smaller_new = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=(2,2), padding="valid",activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(4, 4),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_whole_smaller2 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4,4), strides=(4,4), padding="valid",activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_whole_rgb = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 3)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(4,4), padding="valid",activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(4, 4),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_smaller_rgb = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 3)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(4,4), strides=(4,4), padding="valid",activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_super_small_rgb = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 3)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),
    tf.keras.layers.Conv2D(filters=8, kernel_size=(40,40), strides=(40, 40), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="valid"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(4,4), strides=(4,4), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_super_small = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160, 160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),
    tf.keras.layers.Conv2D(filters=8, kernel_size=(2,2), strides=(2, 2), padding="valid",activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(10, 10), strides=(10, 10), activation='relu', padding="valid"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), activation='relu', padding="valid"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1,activation = 'sigmoid'),
])