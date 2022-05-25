import tensorflow as tf

model_tf = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 243.),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=(2,2), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(4,4), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(5, 5), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])


model_tf3 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 243.),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(8, 8), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(5, 5),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_tf2 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(10,10), strides=(10,10), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(8, 8), strides=(8, 8), padding="valid", activation='relu',),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_simple = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 243.),
    tf.keras.layers.Conv2D(filters=8, kernel_size=(20,20), strides=(20,20), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(8, 8),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

model_simple2 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160, 1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 243.),
    tf.keras.layers.Conv2D(filters=4, kernel_size=(10,10), strides=(10,10), padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=8, kernel_size=(8, 8), strides=(8, 8),  padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])