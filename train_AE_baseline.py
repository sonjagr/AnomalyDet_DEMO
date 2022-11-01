import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
from autoencoders2 import *
from sklearn.metrics import confusion_matrix
from helpers.dataset_helpers import create_cnn_dataset
import random
import matplotlib.pyplot as plt
random.seed(42)
import numpy as np
from helpers.dataset_helpers import create_cnn_dataset, process_anomalous_df_to_numpy
from helpers.cnn_helpers import rotate,rotate_1, rotate_2, rotate_3, bright, format_data, flip_h, flip_v, patch_images, flip, plot_metrics, plot_examples, crop, bright_encode, encode, tf_bayer2rgb
from autoencoders2 import *
from common import *
import random, time, argparse, os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
gpu = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

bce = tf.keras.losses.BinaryCrossentropy()

random.seed(42)
base_dir = TrainDir_gpu
images_dir_loc = imgDir_gpu
dir_det = 'db/DET/'
dir_ae = 'db/AE/'

ae = AutoEncoder()
print('Loading autoencoder and data...')
ae.load(os.path.join(base_dir, 'checkpoints/TQ3_2_1_TQ3_2_more_params_2/AE_TQ3_2_322_to_322_epochs'))

## extract normal images for training
X_list_norm = np.load(os.path.join(base_dir, 'db/AE/NORMAL_TRAIN_20220711.npy'), allow_pickle=True)
print('        Available normal train, val images', len(X_list_norm))

import datetime
f = os.path.join(base_dir, 'db/TRAIN_DATABASE_05_09_2022_2')
with pd.HDFStore( f,  mode='r') as store:
        train_db = store.select('db')
        date = datetime.datetime.strptime('2022-07-25', '%Y-%m-%d').date()
        train_db = train_db[train_db.Date <= date]
        cols = train_db.reset_index().Campaign.unique().tolist()
        print(cols)
        print(f'Reading {f}')
X_train_det_list, Y_train_det_list = process_anomalous_df_to_numpy(train_db)

f = os.path.join(base_dir, 'db/VAL_DATABASE_05_09_2022_2')
with pd.HDFStore( f,  mode='r') as store:
        val_db = store.select('db')
        date = datetime.datetime.strptime('2022-07-25', '%Y-%m-%d').date()
        val_db = val_db[val_db.Date <= date]
        print(f'Reading {f}')
X_val_det_list,  Y_val_det_list = process_anomalous_df_to_numpy(val_db)


f = os.path.join(base_dir, 'db/TEST_DATABASE_05_09_2022_2')
with pd.HDFStore( f,  mode='r') as store:
        test_db = store.select('db')
        date = datetime.datetime.strptime('2022-07-25', '%Y-%m-%d').date()
        test_db = test_db[test_db.Date <= date]
        print(f'Reading {f}')
X_test_det_list,  Y_test_det_list = process_anomalous_df_to_numpy(test_db)

X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_val_det_list = [images_dir_loc + s for s in X_val_det_list]
X_test_det_list = [images_dir_loc + s for s in X_test_det_list]

N_det_val = len(X_val_det_list)
N_det_train = len(X_train_det_list)

np.random.seed(42)

print('    Loaded number of defective train, val samples: ', N_det_train, N_det_val)

## only images with defects
train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list.tolist(), _shuffle=False)
val_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list.tolist(), _shuffle=False)
test_ds = create_cnn_dataset(X_test_det_list, Y_test_det_list.tolist(), _shuffle=False)


@tf.function
def crop(img, lbl):
    img = tf.reshape(img, [-1, 2736, 3840,1])
    img = tf.keras.layers.Cropping2D(cropping=((0, 16), (0, 0)))(img)
    return img, lbl

@tf.function
def encode(img, lbl, ae):
    img = tf.reshape(img, [-1, 2720, 3840, 1])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.math.sqrt(tf.pow(tf.subtract(img, decoded_img), 2))
    return aed_img, lbl

@tf.function
def patch_images(img, lbl):
    split_img = tf.image.extract_patches(images=img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1], rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [17*24, 160 *160])
    lbl = tf.reshape(lbl, [17*24])
    patch_ds = tf.data.Dataset.from_tensors((re, lbl))
    return patch_ds

@tf.function
def process_crop_encode(image, label):
    image, label = crop(image, label)
    image, label = encode(image, label, ae)
    return image,label

@tf.function
def format(image, label):
    image = tf.math.reduce_mean(image)
    label = tf.cast(label, tf.float32)
    return image, label

train_ds = train_ds.map(process_crop_encode)
val_ds = val_ds.map(process_crop_encode)
test_ds = test_ds.map(process_crop_encode)

train_ds = train_ds.flat_map(patch_images).unbatch()
val_ds = val_ds.flat_map(patch_images).unbatch()
test_ds = test_ds.flat_map(patch_images).unbatch()

train_ds = train_ds.map(format)
val_ds = val_ds.map(format)
test_ds = test_ds.map(format)

train_norm = train_ds.filter(lambda x,y: y ==0.)
train_anom = train_ds.filter(lambda x,y: y ==1.)
print(len(list(train_norm)))
print(len(list(train_anom)))
train_norm_means = []
for x, y in train_norm:
    train_norm_means.append(x.numpy())

train_anom_means = []
for x, y in train_anom:
    train_anom_means.append(x.numpy())

print('Anom patches', len(train_anom_means))
print('norm patches', len(train_norm_means))

import sklearn
#train_means = train_norm_means.append(train_anom_means)
#scaler = sklearn.preprocessing.MinMaxScaler()
#train_means = scaler.fit_transform(np.array(train_means).reshape(-1, 1))

#train_norm_means = train_means[:3000]
#train_anom_means =train_means[-len(train_anom_means):]

th = 27.2
plt.hist(train_anom_means,  bins = int(np.sqrt(len(train_anom_means))), density = True, alpha = 0.6, color = 'lightgray', label='Anomalous', zorder = 3)
plt.hist(train_norm_means, bins = int(np.sqrt(len(train_norm_means))),  density = True, alpha = 0.6,color = 'gray', label = 'Normal',zorder = 3)
plt.grid(zorder = 1)
plt.xlabel('Mean pixel-wise reconstruction error [arb. unit]', fontsize = 14)
plt.plot([th, th,th,th], [0.0,0.05,0.2,0.6], linestyle = '--', color = 'black', label='Threshold = 27.2', zorder = 5)
#plt.plot([th, th,th,th], [0.0,4,6,8.6], linestyle = '--', color = 'black', label='Threshold = 0.54', zorder = 5)
plt.ylim(0,0.6)
plt.xlim(15, 45)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(loc = 'upper right', fontsize = 14)
plt.tight_layout()
plt.show()


threshold = 27.2
true = []
pred = []
pred2 = []
true_pos = 0
pred_pos = 0
for x, y in test_ds:
    mean = x
    true.append(y.numpy())
    if y == 1:
        true_pos = true_pos + 1
    if mean < threshold:
        pred.append(0)
        pred2.append(mean)
    if mean > threshold:
        pred.append(1)
        pred2.append(mean)
        pred_pos = pred_pos + 1
print(true_pos)
print(pred_pos)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0,1]).ravel()

cm= confusion_matrix(true, pred, labels=[0,1])
print(cm)
print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

def plot_roc_curve2(fpr1, tpr1, auc1, ):
    plt.figure(2)
    plt.plot(fpr1, tpr1, color = 'C0', label = 'AE baseline, AUC = '+str(round(auc1, 2)))
    plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), linestyle = '--', color = 'gray')
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.ylabel('True Positive Rate', fontsize = 14)
    #plt.xscale('log')
    plt.grid()
    plt.legend(loc = 'lower right', fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    #plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/baseline.png', dpi=600)
    plt.show()
from sklearn.metrics import roc_curve
from sklearn import metrics

fpr1 , tpr1 , thresholds = roc_curve(true, pred2)

auc1 = metrics.auc(fpr1, tpr1)
plt.figure(2)
plot_roc_curve2(fpr1, tpr1, auc1)
print('AUC: ', auc1)
