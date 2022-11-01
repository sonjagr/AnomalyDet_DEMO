import numpy as np
import random, time, argparse, os
import tensorflow as tf

from background_CNNs import *
from common import *
from helpers.dataset_helpers import create_cnn_dataset
from helpers.cnn_helpers import  format_data, patch_images, plot_metrics, plot_examples, crop,  tf_bayer2rgb
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
parser.add_argument("--load", type=bool,
                    help="Load old model or not", default = False, required=False)
parser.add_argument("--rgb", type=bool,
                    help="rgb or bayer", default=False, required=False)

args = parser.parse_args()

num_epochs = args.epochs
batch_size = args.batch_size
model_ID = args.model_ID
gpu = args.gpu
savename = args.savename
load = args.load
lr = args.lr
data_format =args.rgb

if gpu != 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

try:
    os.makedirs('saved_CNNs/br_%s' % savename)
except FileExistsError:
    pass

saveloc = 'saved_CNNs/br_%s' % savename
# write train params to file
f = open(saveloc +'/argfile.txt', "w")
f.write("  Epochs: %s" % num_epochs)
f.write("  Batch_size: %s" % batch_size)
f.write("  Model_ID: %s" % model_ID)
f.write("  Savename: %s" % savename)
f.write("  Lr: %s" % lr)
f.write("  Data format: %s" % data_format)
f.close()

random.seed(42)
base_dir = r"/afs/cern.ch/user/s/sgroenro/anomaly_detection/db"
dir_det = 'DET/'
dir_ae = 'AE/'
images_dir_loc = imgDir_gpu

## extract normal images for training
X_train = np.load(os.path.join(base_dir, dir_det, 'X_train_DET_backround_20220711.npy'), allow_pickle=True)
Y_train = np.load(os.path.join(base_dir, dir_det, 'Y_train_DET_backround_20220711.npy'), allow_pickle=True).tolist()
X_train = [images_dir_loc + s for s in X_train]

X_test = np.load(os.path.join(base_dir, dir_det, 'X_test_DET_backround_20220711.npy'), allow_pickle=True)
Y_test = np.load(os.path.join(base_dir, dir_det, 'Y_test_DET_backround_20220711.npy'), allow_pickle=True).tolist()
X_test = [images_dir_loc + s for s in X_test]

print('All loaded. Starting dataset creation...')

ds = create_cnn_dataset(X_train, Y_train, _shuffle=False).shuffle(1000, seed=42)
test_ds = ds.take(60)
train_ds = ds.skip(60)

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'),
      tf.keras.metrics.Accuracy(name='Acc'),
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

train_ds_nob = train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds_nob = test_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

## apply patching
train_ds = train_ds_nob.flat_map(patch_images).unbatch().cache()
test_ds = test_ds_nob.flat_map(patch_images).unbatch().shuffle(200*408)

train_norm = train_ds.filter(lambda x,y: y ==0.)
train_anom = train_ds.filter(lambda x,y: y ==1.)
print("Number of surface samples: ",len(list(train_norm)))
print("Number of background samples: ", len(list(train_anom)))

train_ds = train_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds_final = train_ds.shuffle(1000*408, reshuffle_each_iteration=True)
test_ds_final = test_ds.cache()

train_ds_batch = train_ds_final.batch(batch_size=batch_size, drop_remainder = True)
test_ds_batch = test_ds_final.batch(batch_size=batch_size, drop_remainder = False)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

if load == False:
    if model_ID == 'br_1':
        model = br_1
if load == True:
    model = tf.keras.models.load_model(os.path.join(saveloc +'/br_cnn_%s' % savename))

model.compile(optimizer = optimizer, loss = tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=METRICS)

print(model.summary())

filepath = saveloc +'/br_cnn_%s' % savename

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

##save training history to file
filename = os.path.join(saveloc, 'history_log.csv')
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

print('Starting training:')
history = model.fit(train_ds_batch.prefetch(tf.data.experimental.AUTOTUNE), epochs = num_epochs, validation_data = test_ds_batch.prefetch(tf.data.experimental.AUTOTUNE), callbacks = [checkpoint_callback, history_logger])
print('Training finished, plotting...')

plot_metrics(history, savename)

