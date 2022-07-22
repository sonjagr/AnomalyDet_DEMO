import numpy as np
from helpers.dataset_helpers import create_cnn_dataset
from helpers.cnn_helpers import rotate, format_data, patch_images, flip, plot_metrics, plot_examples, crop, bright_encode, encode, encode_rgb, bright_encode_rgb, tf_bayer2rgb
from old_codes.autoencoders import *
import random, time, argparse, os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.keras.backend.clear_session()

gpu = '3'

use_ae = False

if gpu != 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

random.seed(42)
base_dir = ''
base_dir = TrainDir_gpu
images_dir_loc = imgDir_gpu
dir_det = '../db/DET/'
dir_ae = '../db/AE/'

if use_ae:
    ae = AutoEncoder()
    print('Loading autoencoder and data...')
    ae.load(os.path.join(base_dir, 'checkpoints/TQ3_2_1_TQ3_2_more_params_2/AE_TQ3_2_277_to_277_epochs'))

## extract normal images for training
X_train_list_ae = np.load(os.path.join(base_dir, dir_ae, 'X_train_NOT_AE.npy'), allow_pickle=True)
X_test_list_ae = np.load(os.path.join(base_dir, dir_ae, 'X_test_NOT_AE.npy'), allow_pickle=True)

## load the images containing anomalies
X_train_det_list = np.load(os.path.join(base_dir, dir_det, 'X_train_DET_final.npy'), allow_pickle=True)
X_val_det_list = np.load(os.path.join(base_dir, dir_det, 'X_test_DET_final.npy'), allow_pickle=True)

X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_val_det_list = [images_dir_loc + s for s in X_val_det_list]

Y_train_det_list = np.load(os.path.join(base_dir, dir_det , 'Y_train_DET_final.npy'), allow_pickle=True).tolist()
Y_val_det_list = np.load(os.path.join(base_dir , dir_det , 'Y_test_DET_final.npy'), allow_pickle=True).tolist()

## split test and validation sets
N_det_val = int(len(X_val_det_list)/2)
N_det_train = len(X_train_det_list)
X_val_det_list = X_val_det_list[:N_det_val]
Y_val_det_list = Y_val_det_list[:N_det_val]

## you can choose up to 4496/1125 whole images
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

shape_for_normal = list(np.shape(Y_train_det_list)[1:])
Y_train_list = np.append(Y_train_det_list, np.full((int(len(X_train_normal_list)), shape_for_normal[0], shape_for_normal[1]), 0.), axis = 0)
Y_val_list = np.append(Y_val_det_list, np.full((int(len(X_val_normal_list)),  shape_for_normal[0], shape_for_normal[1]), 0.), axis = 0)

## only with defects
train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=False)
val_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list, _shuffle=False)

## with normal images
more_normal_train_ds = create_cnn_dataset(X_train_list, Y_train_list, _shuffle=False)
more_normal_val_ds = create_cnn_dataset(X_val_list, Y_val_list, _shuffle=False)

#train_ds = more_normal_train_ds
#val_ds = more_normal_val_ds

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
def process_crop_encode(image, label):
    image, label = crop(image, label)
    image, label = encode(image, label, ae)
    return image, label

@tf.function
def process_crop(image, label):
    image, label = crop(image, label)
    return image, label

if use_ae == True:
    print('    Applying autoencoding')
    print('    Using bayer format')
    train_ds = train_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
if use_ae == False:
    print('    Not applying autoencoding')
    print('    Using bayer format')
    train_ds = train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

## add changes in brighness
np.random.seed(42)

## apply patching
train_ds = train_ds.flat_map(patch_images).unbatch()
val_ds = val_ds.flat_map(patch_images).unbatch()

train_ds_anomaly = train_ds.filter(lambda x, y:  y == 1.)
nbr_anom_patches = len(list(train_ds_anomaly))

print('    Number of anomalous patches: ', nbr_anom_patches)

aug_size = nbr_anom_patches

plot_examples(train_ds_anomaly.shuffle(100).take(3))

rotations = np.random.randint(low = 1, high = 4, size = aug_size).astype('int32')
counter2 = tf.data.Dataset.from_tensor_slices(rotations)
train_ds_to_rotate = tf.data.Dataset.zip((train_ds_anomaly, counter2))
rotated = train_ds_to_rotate.map(rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#plot_examples(rotated.skip(10).take(3))

flip_seeds = np.random.randint(low = 1, high = 11, size = aug_size).astype('int32')
counter3 = tf.data.Dataset.from_tensor_slices(flip_seeds)
train_ds_to_flip = tf.data.Dataset.zip((train_ds_anomaly, counter3))
flipped = train_ds_to_flip.map(flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#plot_examples(flipped.skip(10).take(3))

train_ds_rotated = train_ds_anomaly.concatenate(rotated)
train_ds_rotated_flipped = train_ds_rotated.concatenate(flipped)

augmented = train_ds_rotated_flipped.cache()
anomalous = nbr_anom_patches*3

train_ds_normal_all = train_ds.filter(lambda x,y: y == 0.)
plot_examples(train_ds_normal_all.skip(10).take(3))
normal = N_det_train*408-nbr_anom_patches
train_ds_normal_all = train_ds_normal_all

mean_error = 27

train_ds_normal_l = train_ds_normal_all.filter(lambda x, y: tf.reduce_mean(x) > mean_error)

#plot_examples(train_ds_normal_all.take(30))
train_ds_final = train_ds_normal_all.concatenate(augmented).cache()
#plot_examples(augmented.take(30))
train_ds_final = train_ds_final.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds_final = val_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
##plot_examples(val_ds_final.filter(lambda x, y: y == 0.).take(10))
#plot_examples(val_ds_final.filter(lambda x, y: y == 1.).take(10))

print('    Number of normal, anomalous samples: ', normal, anomalous)
print('    Anomaly weight in data: ', normal/anomalous)

