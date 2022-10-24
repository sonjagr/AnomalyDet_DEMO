import tensorflow as tf
import os

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

def VGG16_larger():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(160, 160, 1)))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
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

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(64, activation='relu', name='fc2'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='output'))
    return model


def VGG16_small():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(160, 160, 1)))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name='vgg16'))

    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(64, activation='relu', name='fc1'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation='relu', name='fc2'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='output'))
    return model

def VGG16_small2():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(160, 160, 1)))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.))
    model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name='vgg16'))

    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(64, activation='relu', name='fc1'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(32, activation='relu', name='fc2'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='output'))
    return model


def VGG16_small_do_bn():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(160, 160, 1)))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name='vgg16'))

    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(128, activation='relu', name='fc1'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(64, activation='relu', name='fc2'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation='relu', name='fc3'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='output'))
    return model

def VGG16_small2_do():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(160, 160, 1)))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
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
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu', name='fc2'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='output'))
    return model


def AlexNet():
    AlexNet = tf.keras.Sequential()
    AlexNet.add(tf.keras.layers.Conv2D(filters=96, input_shape=(160, 160, 1), kernel_size=(11, 11), strides=(4, 4), padding='same'))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(tf.keras.layers.Activation('relu'))
    AlexNet.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 2nd Convolutional Layer
    AlexNet.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(tf.keras.layers.Activation('relu'))
    AlexNet.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 3rd Convolutional Layer
    AlexNet.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(tf.keras.layers.Activation('relu'))

    # 4th Convolutional Layer
    AlexNet.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(tf.keras.layers.Activation('relu'))

    # 5th Convolutional Layer
    AlexNet.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(tf.keras.layers.Activation('relu'))
    AlexNet.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Passing it to a Fully Connected layer
    AlexNet.add(tf.keras.layers.Flatten())
    # 1st Fully Connected Layer
    AlexNet.add(tf.keras.layers.Dense(4096, input_shape=(32, 32, 3,)))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(tf.keras.layers.Activation('relu'))
    # Add Dropout to prevent overfitting
    AlexNet.add(tf.keras.layers.Dropout(0.4))

    # 2nd Fully Connected Layer
    AlexNet.add(tf.keras.layers.Dense(4096))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(tf.keras.layers.Activation('relu'))
    # Add Dropout
    AlexNet.add(tf.keras.layers.Dropout(0.4))

    # 3rd Fully Connected Layer
    AlexNet.add(tf.keras.layers.Dense(1000))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(tf.keras.layers.Activation('relu'))
    # Add Dropout
    AlexNet.add(tf.keras.layers.Dropout(0.4))

    # Output Layer
    AlexNet.add(tf.keras.layers.Dense(10))
    AlexNet.add(tf.keras.layers.BatchNormalization())
    AlexNet.add(tf.keras.layers.Activation('sigmoid'))
    return AlexNet






