import numpy as np
INPUT_DIM = 3
from helpers.dataset_helpers import create_cnn_dataset
from helpers.cnn_helpers import  format_data, patch_images, , plot_metrics, plot_examples, crop,  tf_bayer2rgb
import random, time, argparse, os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--model_ID", type=str, help="Model ID",
                    default="model", required=True)
parser.add_argument("--batch_size", type=int,
                    help="Batch size for the training", default=256, required=False)
parser.add_argument("--epochs", type=int,
                    help="Number of epochs for the training", default=50, required=False)
parser.add_argument("--lr", type=float,
                    help="Learning rate", default=1e-3, required=False)
parser.add_argument("--gpu", type=str,
                    help="Which gpu to use", default="1", required=False)
parser.add_argument("--savename", type=str,
                    help="Name to save with", required=True)
parser.add_argument("--load", type=str,
                    help="Load old model or not", default = False, required=False)
parser.add_argument("--contfromepoch", type=bool,
                    help="Epoch to continue training from", default=1, required=False)
parser.add_argument("--data_format", type=bool,
                    help="rgb or bayer", default='bayer', required=False)

args = parser.parse_args()

num_epochs = args.epochs
batch_size = args.batch_size
model_ID = args.model_ID
gpu = args.gpu
savename = args.savename
load = args.load
lr = args.lr
cont_epoch = args.contfromepoch
data_format = args.data_format


if gpu != 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

try:
    os.makedirs('saved_CNNs/br_%s' % savename)
except FileExistsError:
    pass
saveloc = 'saved_CNNs/br_%s' % savename
f = open(saveloc +'/argfile.txt', "w")
f.write("  Epochs: %s" % num_epochs)
f.write("  Batch_size: %s" % batch_size)
f.write("  Model_ID: %s" % model_ID)
f.write("  Savename: %s" % savename)
f.write("  Lr: %s" % lr)
f.write("  Data format: %s" % data_format)
f.close()

random.seed(42)
from common import *
base_dir = DataBaseFileLocation_local
dir_det = 'DET/'
dir_ae = 'AE/'
images_dir_loc = imgDir_pc

## extract normal images for training
X = np.load(os.path.join(base_dir, dir_ae, 'X_bgrnd.npy'), allow_pickle=True)
Y = np.load(os.path.join(base_dir, dir_ae, 'Y_bgrnd.npy'), allow_pickle=True).tolist()
print(Y)
X = [images_dir_loc + s for s in X]

print('All loaded. Starting dataset creation...')

ds = create_cnn_dataset(X, Y, _shuffle=False)

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'),
]

@tf.function
def process_crop(image, label):
    image, label = crop(image, label)
    return image, label

@tf.function
def process_crop_rgb(image, label):
    image, label = crop(image, label)
    image = tf_bayer2rgb(image)
    return image, label

print('    Not applying autoencoding')

train_ds_nob = ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

## apply patching
ds = train_ds_nob.flat_map(patch_images).unbatch().shuffle(50*408)

test_ds = ds.take(4080)
train_ds = ds.skip(4080)

train_ds = train_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds_final = train_ds.cache()
test_ds_final = test_ds.cache()

train_ds_batch = train_ds_final.batch(batch_size=batch_size, drop_remainder = True)
test_ds_batch = test_ds_final.batch(batch_size=batch_size, drop_remainder = False)

#plot_examples(train_ds_anomaly.skip(2).take(5))
#plot_examples(train_ds_normal.skip(2).take(5))
#plot_examples(val_ds.take(5))

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

from backround_CNNs import *
if model_ID == 'br_1':
    model = br_1

model.compile(optimizer = optimizer, loss = tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=METRICS)

print(model.summary())

filepath = saveloc +'/br_cnn_%s' % savename

##save every 10th epoch
save_every = int(np.floor((len(list(train_ds)))/batch_size)*5)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

##save training history to file
filename = saveloc +'/history_log.csv'
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

print('Starting training:')
history = model.fit(train_ds_batch.prefetch(tf.data.experimental.AUTOTUNE), epochs = num_epochs, validation_data = test_ds_batch.prefetch(tf.data.experimental.AUTOTUNE), callbacks = [checkpoint_callback, history_logger])
print('Training finished, plotting...')

plot_metrics(history, savename)

