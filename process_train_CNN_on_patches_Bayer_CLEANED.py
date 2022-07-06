import numpy as np
from helpers.dataset_helpers import create_cnn_dataset
from helpers.cnn_helpers import rotate, bright, format_data, patch_images, flip, plot_metrics, plot_examples, crop, bright_encode, encode, tf_bayer2rgb
from old_codes.autoencoders import *
import random, time, argparse, os
import tensorflow as tf
import pandas as pd
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
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
parser.add_argument("--load", type=int,
                    help="Load old model or not", default = 0, required=False)
parser.add_argument("--contfromepoch", type=int,
                    help="Epoch to continue training from", default=1, required=False)
parser.add_argument("--use_ae", type=int,
                    help="use ae or not", default=1, required=False)
parser.add_argument("--take_normal", type=int,
                    help="how many times normal", default=10, required=False)
parser.add_argument("--anomaly_weight", type=int,
                    help="anomaly_weight", default=2, required=False)
parser.add_argument("--bright_aug", type=int,
                    help="augment brightness or not", default=1, required=False)

args = parser.parse_args()

num_epochs = args.epochs
batch_size = args.batch_size
model_ID = args.model_ID
gpu = args.gpu
savename = args.savename
load = args.load
lr = args.lr
cont_epoch = args.contfromepoch
use_ae = args.use_ae
take_normal = args.take_normal
anomaly_weight = args.anomaly_weight
bright_aug = args.bright_aug

def write_file():
    f = open('saved_CNNs/%s/argfile.txt' % savename, "w")
    f.write("  Epochs: %s" % num_epochs)
    f.write("  Batch_size: %s" % batch_size)
    f.write("  Model_ID: %s" % model_ID)
    f.write("  Savename: %s" % savename)
    f.write("  Lr: %s" % lr)
    f.write("  AE: %s" % use_ae)
    f.write("  Nbr of times normal: %s" % take_normal)
    f.write("  Anomaly_weight: %s" % anomaly_weight)
    f.close()

if gpu != 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

try:
    os.makedirs('saved_CNNs/%s' % savename)
except FileExistsError:
    pass

random.seed(42)
base_dir = TrainDir_gpu
images_dir_loc = imgDir_gpu
dir_det = 'db/DET/'
dir_ae = 'db/AE/'

if use_ae == 1:
    ae = AutoEncoder()
    print('Loading autoencoder and data...')
    ae.load(os.path.join(base_dir, 'checkpoints/TQ3_2_1_TQ3_2_more_params_2/AE_TQ3_2_200_to_200_epochs'))

## extract normal images for training
X_train_list_ae = np.load(os.path.join(base_dir, dir_ae, 'X_train_forcnn_clean.npy'), allow_pickle=True)
X_test_list_ae = np.load(os.path.join(base_dir, dir_ae, 'X_test_forcnn_clean.npy'), allow_pickle=True)
X_train_list_ae = [s.replace('F:/ScratchDetection/MeasurementCampaigns/', '') for s in X_train_list_ae]
X_test_list_ae = [s.replace('F:/ScratchDetection/MeasurementCampaigns/', '') for s in X_test_list_ae]

f = os.path.join(base_dir, 'db/train_defective_df.h5')
with pd.HDFStore( f,  mode='r') as store:
        train_val_db = store.select('db')
        print(f'Reading {DataBaseFileLocation_local+f}')
train_df = train_val_db.sample(frac=0.89)
val_df = train_val_db.drop(train_df.index)

X_train_det_list = train_df['Path'].to_numpy()
X_val_det_list = val_df['Path'].to_numpy()

Y_train_det_list = train_df['crop_lbls'].to_numpy().flatten()
Y_val_det_list = val_df['crop_lbls'].to_numpy().flatten()

X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_val_det_list = [images_dir_loc + s for s in X_val_det_list]

## split test and validation sets
N_det_val = len(X_val_det_list)
N_det_train = len(X_train_det_list)

## you can choose up to 2000/420 whole images
np.random.seed(42)
X_train_normal_list = np.random.choice(X_train_list_ae, int(N_det_train), replace=False)
X_val_normal_list = np.random.choice(X_test_list_ae, int(N_det_val), replace=False)
X_train_normal_list = [images_dir_loc + s for s in X_train_normal_list]
X_val_normal_list = [images_dir_loc + s for s in X_val_normal_list]

print('    Loaded number of train, val samples: ', N_det_train, N_det_val)
print('All loaded. Starting dataset creation...')
time1 = time.time()

## combine anomalous and normal images
X_train_list = np.append(X_train_det_list, X_train_normal_list, axis = 0)
X_val_list = np.append(X_val_det_list, X_val_normal_list, axis = 0)

## only with defects
train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list.tolist(), _shuffle=False)
val_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list.tolist(), _shuffle=False)

## with normal images
#more_normal_train_ds = create_cnn_dataset(X_train_list, Y_train_list, _shuffle=False)
#more_normal_val_ds = create_cnn_dataset(X_val_list, Y_val_list, _shuffle=False)

normal_train_ds = create_cnn_dataset(X_train_normal_list, np.full((int(len(X_train_normal_list)), 408), 0.))
#error_only_normal_ds = normal_train_ds.filter(lambda x, y: tf.reduce_mean(x) > 27)

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

if load == 1:
    print('    Loading previously trained model...')
    model = tf.keras.models.load_model(os.path.join(base_dir, 'saved_CNNs/%s/cnn_%s_epoch_%s' % (savename, savename, cont_epoch)))
if load == 0:
    print('    Not loading an old model...')
    from new_CNNs import *
    if model_ID == 'vgg':
        model = VGG16()

@tf.function
def process_crop_encode(image, label):
    image, label = crop(image, label)
    image, label = encode(image, label, ae)
    return image, label

@tf.function
def process_crop_bright_encode(image_label, delta):
    image, label = image_label
    image, label = crop(image, label)
    image, label = bright_encode(image, label, ae, delta)
    return image, label

@tf.function
def process_crop(image, label):
    image, label = crop(image, label)
    return image, label

@tf.function
def process_crop_bright(image, label, delta):
    image, label = crop(image, label)
    image, label = bright_encode(image, label, ae, delta)
    return image, label

def filter_anom(x):
  return tf.math.equal(x, 1)

def filter_norm(x):
  return tf.math.equal(x, 0)

np.random.seed(42)
train_ds_orig = train_ds

if use_ae == 1:
    print('    Applying autoencoding')
    print('    Using bayer format')
    train_ds = train_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    normal_train_ds = normal_train_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
elif use_ae == 0:
    print('    Not applying autoencoding')
    print('    Using bayer format')
    train_ds = train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    normal_train_ds = normal_train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

## add changes in brighness
np.random.seed(42)

if bright_aug == 1 and use_ae == 1:
    print('    Augmenting brightness of images')
    brightnesses = np.random.uniform(low = 0.75, high = 1.25, size = np.array(Y_train_det_list).shape[0])
    counter1 = tf.data.Dataset.from_tensor_slices(brightnesses)
    train_ds_to_bright = tf.data.Dataset.zip((train_ds_orig, counter1))
    train_ds_brightness = train_ds_to_bright.map(lambda x, z: process_crop_bright_encode(x, z), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds_brightness = train_ds_brightness.flat_map(patch_images).unbatch()

## apply patching
train_ds = train_ds.flat_map(patch_images).unbatch()
normal_train_ds = normal_train_ds.flat_map(patch_images).unbatch()
val_ds = val_ds.flat_map(patch_images).unbatch()

train_ds_anomaly = train_ds.filter(lambda x, y:  filter_anom(y))
plot_examples(train_ds_anomaly.skip(10).take(5))

nbr_anom_patches = len(list(train_ds_anomaly))

if bright_aug == 1:
    train_ds_brightness_anomaly = train_ds_brightness.filter(lambda x, y: filter_anom(y))
    train_ds_anomaly = train_ds_anomaly.concatenate(train_ds_brightness_anomaly)

print('    Number of anomalous patches: ', nbr_anom_patches)

if bright_aug == True:
    aug_size = 2*nbr_anom_patches
if bright_aug == False:
    aug_size = nbr_anom_patches

rotations = np.random.randint(low = 1, high = 4, size = aug_size).astype('int32')
counter2 = tf.data.Dataset.from_tensor_slices(rotations)
train_ds_to_rotate = tf.data.Dataset.zip((train_ds_anomaly, counter2))

flip_seeds = np.random.randint(low = 1, high = 11, size = aug_size).astype('int32')
counter3 = tf.data.Dataset.from_tensor_slices(flip_seeds)
train_ds_to_flip = tf.data.Dataset.zip((train_ds_anomaly, counter3))

train_ds_rotated = train_ds_anomaly.concatenate(train_ds_to_rotate.map(rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE))
train_ds_rotated_flipped = train_ds_rotated.concatenate(train_ds_to_flip.map(flip, num_parallel_calls=tf.data.experimental.AUTOTUNE))

augmented = train_ds_rotated_flipped

train_ds_normal_all = normal_train_ds
normal = len(list(train_ds_normal_all))
anomalous = aug_size*3
#normal = N_det_train*408-nbr_anom_patches

val_ds_normal = val_ds.filter(lambda x, y: filter_norm(y)).shuffle(seed=42).cache()
val_ds_anom = val_ds.filter(lambda x, y: filter_anom(y))

val_ds_final = val_ds_anom.concatenate(val_ds_normal.take(len(list(val_ds_anom)))*(normal/anomalous))

train_ds_final = train_ds_normal_all.concatenate(augmented).cache()

train_ds_final = train_ds_final.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds_final = val_ds_final.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

print('    Number of normal, anomalous samples: ', normal, anomalous)
print('    Anomaly weight set by user, anomaly weight in data: ', anomaly_weight, normal/anomalous)

train_ds_final = train_ds_final.shuffle(buffer_size=normal+anomalous, reshuffle_each_iteration=True)
val_ds_final = val_ds_final.cache()

train_ds_batch = train_ds_final.batch(batch_size=batch_size, drop_remainder = True)
val_ds_batch = val_ds_final.batch(batch_size=batch_size, drop_remainder = False)

plot_examples(train_ds_normal_all.skip(5).take(5))

time2 = time.time()
pro_time = time2-time1
print('Dataset (time {:.2f} s) created, starting training...'.format(pro_time))

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
fl = SigmoidFocalCrossEntropy(gamma= 2, reduction=tf.keras.losses.Reduction.AUTO)
model.compile(optimizer = optimizer, loss = fl, metrics=METRICS)

print(model.summary())

filepath = 'saved_CNNs/%s/cnn_%s_epoch_{epoch:02d}' % (savename, savename)
save_every = int(np.floor((normal+anomalous)/batch_size)*2)
filepath_loss = 'saved_CNNs/%s/cnn_%s_epoch_{epoch:02d}' % (savename, savename)
#checkpoint_callback_best_tp = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_tp, monitor='val_tp', mode='max', verbose=1, save_best_only=False, save_every = save_every)
checkpoint_callback_best_loss = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_loss, monitor='val_loss', mode='min', verbose=1, save_best_only=False, save_every = save_every)

##save training history to file
filename = 'saved_CNNs/%s/history_log.csv' % savename
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

##add lr schedule
#lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

write_file()

print('Starting training:')
history = model.fit(train_ds_batch.prefetch(1), epochs = num_epochs, validation_data = val_ds_batch.prefetch(1), callbacks = [checkpoint_callback_best_loss, history_logger])
print('Training finished, plotting...')

plot_metrics(history, savename)

