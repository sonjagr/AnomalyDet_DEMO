import os
import tensorflow as tf
from common import *

# define the encoder and decoder networks
class AutoEncoder(tf.keras.Model):
    """Convolutional DNN-based autoencoder."""

    def __init__(self):
        super(AutoEncoder, self).__init__()

    # original with less filters
    def initialize_network_TQ1(self):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    PICTURESIZE_Y, PICTURESIZE_X, 1)),
                tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(8, 8), strides=(8, 8), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(2, 2), strides=(2, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=18, kernel_size=(3, 5), strides=(3, 5), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=18, kernel_size=(3, 2), strides=(3, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=12, kernel_size=(1, 1), strides=(1, 1), activation='elu', padding="valid", use_bias=True),
            ]
        )
        print(self.encoder.summary())

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.encoder.output.shape[1:]),
                tf.keras.layers.Conv2D(
                    filters=12, kernel_size=(1, 1), strides=(1, 1), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=12, kernel_size=(3, 2), strides=(3, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(3, 5), strides=(3, 5), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(2, 2), strides=(2, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(8, 8), strides=(8, 8), activation='relu', padding="valid", use_bias=True),

                tf.keras.layers.experimental.preprocessing.Rescaling(EIGHTBITMAX),
            ]
        )
        print(self.decoder.summary())

    ### original
    def initialize_network_TQ2(self):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    2736,PICTURESIZE_X, 1)),
                tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(8, 8), strides=(8, 8), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(2, 2), strides=(2, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(3, 5), strides=(3, 5), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(3, 2), strides=(3, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(1, 1), strides=(1, 1), activation='elu', padding="valid", use_bias=True),
            ]
        )
        print(self.encoder.summary())

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.encoder.output.shape[1:]),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(1, 1), strides=(1, 1), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(3, 2), strides=(3, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(3, 5), strides=(3, 5), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(2, 2), strides=(2, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(8, 8), strides=(8, 8), activation='relu', padding="valid", use_bias=True),

                tf.keras.layers.experimental.preprocessing.Rescaling(EIGHTBITMAX),
            ]
        )
        print(self.decoder.summary())

    ## crops into 17x24
    def initialize_network_TQ3(self):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    2720, PICTURESIZE_X, 1)),
                tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(10, 8), strides=(10, 8), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(4, 2), strides=(4, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(2, 5), strides=(2, 5), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(2, 2), strides=(2, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(1, 1), strides=(1, 1), activation='elu', padding="valid", use_bias=True),
            ]
        )
        print(self.encoder.summary())

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.encoder.output.shape[1:]),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(1, 1), strides=(1, 1), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(2, 2), strides=(2, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(2, 5), strides=(2, 5), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(4, 2), strides=(4, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(10, 8), strides=(10, 8), activation='relu', padding="valid", use_bias=True),

                tf.keras.layers.experimental.preprocessing.Rescaling(EIGHTBITMAX),
            ]
        )
        print(self.decoder.summary())

    # decreased layers and filters, does not follow cropping
    def initialize_network_SG1(self):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    2736, PICTURESIZE_X, 1)),
                tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(6, 4), strides=(6, 4), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(8, 4), strides=(8, 4), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(3, 8), strides=(3, 8), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(1, 1), strides=(1, 1), activation='elu', padding="valid",
                    use_bias=True),
            ]
        )
        print(self.encoder.summary())

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.encoder.output.shape[1:]),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(1, 1), strides=(1, 1), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=(3, 8), strides=(3, 8), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=(8, 4), strides=(8, 4), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(6, 4), strides=(6, 4), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="valid",
                    use_bias=True),
                tf.keras.layers.experimental.preprocessing.Rescaling(EIGHTBITMAX),
            ]
        )
        print(self.decoder.summary())

    # relu, less filters, layers, does the cropping right
    def initialize_network_SG2(self):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    2720, PICTURESIZE_X, 1)),
                tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
                tf.keras.layers.Conv2D(
                    filters=18, kernel_size=(10, 10), strides=(10, 10), activation='relu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(8, 8), strides=(8, 8), activation='relu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="valid",
                    use_bias=True),
            ]
        )
        print(self.encoder.summary())

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.encoder.output.shape[1:]),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(8, 8), strides=(8, 8), activation='relu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=18, kernel_size=(10, 10), strides=(10, 10), activation='relu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="valid",
                    use_bias=True),
                tf.keras.layers.experimental.preprocessing.Rescaling(EIGHTBITMAX),
            ]
        )
        print(self.decoder.summary())

    # selu, overlapping, smaller latent space
    def initialize_network_SG3(self):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    2720, PICTURESIZE_X, 1)),
                tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(8, 10), strides=(6, 5), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(8, 7), strides=(5, 8), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(6, 2), strides=(4, 2), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(2, 3), strides=(2, 3), activation='elu', padding="valid",
                    use_bias=True),
            ]
        )
        print(self.encoder.summary())

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.encoder.output.shape[1:]),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(2, 3), strides=(2, 3), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(6, 2), strides=(4, 2), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=(8, 7), strides=(5, 8), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=(8, 10), strides=(6, 5), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding="valid",
                    use_bias=True),
                tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 5))),
                tf.keras.layers.experimental.preprocessing.Rescaling(EIGHTBITMAX),
            ]
        )
        print(self.decoder.summary())

    # selu, overlapping, smaller latent space
    def initialize_network_SG4(self):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    2720, PICTURESIZE_X, 1)),
                tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(3, 3), strides=(2, 2), activation='elu', padding="same",
                    use_bias=True),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=None),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(4, 4), strides=(4, 4), activation='elu', padding="same",
                    use_bias=True),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None),
            ]
        )
        print(self.encoder.summary())

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.encoder.output.shape[1:]),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=(4, 4), strides=(4, 4), activation='elu', padding="same",
                    use_bias=True),
                tf.keras.layers.UpSampling2D(2),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(3, 3), strides=(2, 2), activation='elu', padding="same",
                    use_bias=True),
                tf.keras.layers.UpSampling2D(2),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same",
                    use_bias=True),
                #tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 5))),
                tf.keras.layers.experimental.preprocessing.Rescaling(EIGHTBITMAX),
            ]
        )
        print(self.decoder.summary())

    # selu, overlapping, smaller latent space
    def initialize_network_SG5(self):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    2720, PICTURESIZE_X, 1)),
                tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(8, 10), strides=(6, 5), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(8, 7), strides=(5, 8), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(6, 2), strides=(4, 2), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(2, 3), strides=(2, 3), activation='elu', padding="valid",
                    use_bias=True),
            ]
        )
        print(self.encoder.summary())

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.encoder.output.shape[1:]),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=(2, 3), strides=(2, 3), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=(6, 2), strides=(4, 2), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(8, 7), strides=(5, 8), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=2, kernel_size=(8, 10), strides=(6, 5), activation='elu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="valid",
                    use_bias=True),
                tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 5))),
                tf.keras.layers.experimental.preprocessing.Rescaling(EIGHTBITMAX),
            ]
        )
        print(self.decoder.summary())

    def encode(self, im):
        return self.encoder(im)

    def decode(self, z):
        return self.decoder(z)

    def save(self, path):
        encoder_path = path + "_encoder"
        self.encoder.save(encoder_path)

        decoder_path = path + "_decoder"
        self.decoder.save(decoder_path)

    def load(self, fpath):
        encoder_path = fpath + "_encoder"
        if os.path.exists(encoder_path):
            self.encoder = tf.keras.models.load_model(encoder_path, compile=True)
        else:
            return False

        decoder_path = fpath + "_decoder"
        self.decoder = tf.keras.models.load_model(decoder_path, compile=True)
        print("Loaded network from", fpath)
        return True