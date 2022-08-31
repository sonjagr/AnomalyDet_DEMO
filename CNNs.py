import os
import tensorflow as tf
from common import *

# define the encoder and decoder networks
class CNN(tf.keras.Model):

    def __init__(self):
        super(CNN, self).__init__()

    def initialize_network_TQ3(self):
        self.model =

    def save(self, path):
        encoder_path = path + "_encoder"
        self.encoder.save(encoder_path)

        decoder_path = path + "_decoder"
        self.decoder.save(decoder_path)

    def load(self, fpath):
        if os.path.exists(fpath):
            self.encoder = tf.keras.models.load_model(fpath, compile=True)
        else:
            return False
        print("Loaded network from", fpath)
        return True

    def predict(self, im):
        return self.model(im)
