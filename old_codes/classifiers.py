PICTURESIZE_Y = 2748-12
PICTURESIZE_X = 3840

import tensorflow as tf

class AnomalyDetector(tf.keras.Model):
    """Convolutional DNN-based image classification."""

    def __init__(self, dropout=0.2):
        super(AnomalyDetector, self).__init__()
        self.dropout = dropout

    def initialize_cropping_network(self, input_depth=3):
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomContrast(0.2),
        ])

        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    PICTURESIZE_Y, PICTURESIZE_X, input_depth)),
                data_augmentation,
                tf.keras.layers.experimental.preprocessing.Rescaling(1./255.),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(5, 4), strides=(5, 4), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(4, 4), strides=(4, 4), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=(2, 5), strides=(2, 5), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=(4, 2), strides=(4, 2), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding="valid")
            ]
        )
        print(self.cnn.summary())

    def initialize_single_shot_network(self, input_depth=3):
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    PICTURESIZE_Y, PICTURESIZE_X, input_depth)),
                tf.keras.layers.experimental.preprocessing.Rescaling(1./255.),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(5, 4), strides=(5, 4), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(4, 4), strides=(4, 4), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=(2, 5), strides=(2, 5), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=(4, 2), strides=(4, 2), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding="valid")
            ]
        )
        print(self.cnn.summary())

    def initialize_bounding_box_network(self, input_depth=3):
        self.cnn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    PICTURESIZE_Y, PICTURESIZE_X, input_depth)),
                tf.keras.layers.experimental.preprocessing.Rescaling(1./255.),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(5, 4), strides=(5, 4), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(4, 4), strides=(4, 4), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=(2, 5), strides=(2, 5), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=(4, 2), strides=(4, 2), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding="valid")
            ]
        )
        print(self.cnn.summary())

    @tf.function
    def evaluate(self, im):
        return self.cnn(im)

    def save(self, path):
        return self.cnn.save(path)

    def load(self, fpath):
        self.cnn = tf.keras.models.load_model(fpath)
