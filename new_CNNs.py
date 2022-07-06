import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

'''
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
'''
def model_whole_final_overlap():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(160, 160, 1)))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2), strides=(1,1), padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(8, 8), strides=(2, 2), padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2),  padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(2, 2), strides=(1, 1), padding="valid", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    return model

def VGG16():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(160, 160, 1)))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.))
    model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name='vgg16'))
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(128, activation='relu', name='fc1'))
    model.add(tf.keras.layers.Dense(64, activation='relu', name='fc2'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='output'))
    return model


'''
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
'''


