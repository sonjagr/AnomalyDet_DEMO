import numpy as np
import os
INPUT_DIM = 3
from helpers.dataset_helpers import create_cnn_dataset
from helpers.cnn_helpers import rotate, format_data, bayer2rgb, tf_bayer2rgb, patch_images, flip, plot_metrics, plot_examples, crop, bright_encode_rgb, encode, encode_rgb
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
import random, time, argparse
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.keras.backend.clear_session()

num_epochs = 1
batch_size = 1
gpu = '3'
load = False
use_ae = True
bright_aug = True

if gpu != 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

random.seed(42)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

if use_ae:
    ae = AutoEncoder()
    print('Loading autoencoder and data...')
    ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_TQ3_more_data/AE_TQ3_400_to_400_epochs')
    #ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
dir_ae = 'AE/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

## extract normal images for training
X_train_list_ae = np.load(base_dir + dir_ae + 'X_train_AE.npy', allow_pickle=True)
X_test_list_ae = np.load(base_dir + dir_ae + 'X_test_AE.npy', allow_pickle=True)

np.random.seed(1)
X_train_list_normal_to_remove = np.random.choice(X_train_list_ae, 16000, replace=False)

X_train_list_normal_removed = [x for x in X_train_list_ae if x not in X_train_list_normal_to_remove]

## load the images containing anomalies
X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET_2.npy', allow_pickle=True)
X_train_det_list = [images_dir_loc + s for s in X_train_det_list]

Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET_2.npy', allow_pickle=True).tolist()

## split test and validation sets
N_det_train = len(X_train_det_list)

X_train_normal_list = np.random.choice(X_train_list_normal_removed, int(N_det_train), replace=False)
X_train_normal_list = [images_dir_loc + s for s in X_train_normal_list]

print('    Loaded number of train, val samples: ', N_det_train)
print('All loaded. Starting dataset creation...')
time1 = time.time()

## combine anomalous and normal images
X_train_list = np.append(X_train_det_list, X_train_normal_list)

shape_for_normal = list(np.shape(Y_train_det_list)[1:])
Y_train_list = np.append(Y_train_det_list, np.full((int(len(X_train_normal_list)), shape_for_normal[0], shape_for_normal[1]), 0.), axis = 0)

## only with defects
train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=False)

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
    return image,label

@tf.function
def process_crop(image, label):
    image, label = crop(image, label)
    image = tf_bayer2rgb(image)
    return image,label

if use_ae == True:
    print('    Applying autoencoding')
    train_ds_nob = train_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
if use_ae == False:
    print('    Not applying autoencoding')
    train_ds_nob = train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

## add changes in brighness
np.random.seed(42)

## apply patching
train_ds = train_ds_nob.flat_map(patch_images).unbatch()

nbr_anom_patches = len(list(train_ds.filter(lambda x, y:  y == 1.)))

train_ds_normal = train_ds.filter(lambda x,y: y == 0.)

train_ds_normal = train_ds_normal.map(lambda x, y: (tf.reduce_mean(x), y))

print('Filtering out easy normal patches, this might take a while...')
new_normals = []
means = []
indx = 0
for img, lbl in train_ds_normal:
    img_mean = img.numpy()
    means = np.append(means, img_mean)
    indx = indx + 1
    if indx % 1000 == 0:
        print(indx, ' processed')
print('MEAN: ', np.mean(means))

