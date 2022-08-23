import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
from old_codes.autoencoders import *
from sklearn.metrics import confusion_matrix
from helpers.dataset_helpers import create_cnn_dataset
import random
import matplotlib.pyplot as plt
random.seed(42)

gpu = "3"

#os.environ["CUDA_VISIBLE_DEVICES"] = gpu

bce = tf.keras.losses.BinaryCrossentropy()

ae = AutoEncoder()
print('Loading autoencoder and data...')
#ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')
#ae.load('saved_models/model_AE_TQ3_500_to_500_epochs')
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_TQ3_more_data/AE_TQ3_318_to_318_epochs')

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
#base_dir = 'db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
#images_dir_loc = 'F:/ScratchDetection/MeasurementCampaigns/'

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_test = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)
X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_test = [images_dir_loc + s for s in X_test]
Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET.npy', allow_pickle=True).tolist()
Y_test = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()

N_det_val = int(len(X_test)/2)
X_val_det_list = X_test[:N_det_val]
Y_val_det_list = Y_test[:N_det_val]

X_test_det_list = X_test[-N_det_val:]
Y_test_det_list = Y_test[-N_det_val:]

N_det_train = len(X_train_det_list)
print('Loaded number of train, val samples: ', N_det_train, N_det_val)
print('All loaded. Starting processing...')

train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=True)
val_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list, _shuffle=True)
test_ds = create_cnn_dataset(X_test_det_list, Y_test_det_list, _shuffle=True)

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

train_norm = train_ds.filter(lambda x,y: y ==0.).take(3000)
train_anom = train_ds.filter(lambda x,y: y ==1.)

train_norm_means = []
for x, y in train_norm:
    train_norm_means.append(x.numpy())

train_anom_means = []
for x, y in train_anom:
    train_anom_means.append(x.numpy())

import sklearn
train_means = train_norm_means.append(train_anom_means)
scaler = sklearn.preprocessing.MinMaxScaler()
train_means = scaler.fit_transform(np.array(train_means).reshape(-1, 1))

train_norm_means = train_means[:3000]
train_anom_means =train_means[-len(train_anom_means):]

th = 0.54
plt.hist(train_anom_means,  bins = int(np.sqrt(len(train_anom_means))), density = True, alpha = 0.6, color = 'lightgray', label='Anomalous', zorder = 3)
plt.hist(train_norm_means, bins = int(np.sqrt(len(train_norm_means))),  density = True, alpha = 0.6,color = 'gray', label = 'Non-anomalous',zorder = 3)
plt.grid(zorder = 1)
plt.xlabel('Mean pixel-wise reconstruction error', fontsize = 14)
plt.plot([th, th,th,th], [0.0,0.05,0.2,0.6], linestyle = '--', color = 'black', label='Threshold = 27.1')
#plt.plot([th, th,th,th], [0.0,4,6,8.6], linestyle = '--', color = 'black', label='Threshold = 0.54', zorder = 5)
#plt.ylim(0,8.5)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(loc = 'upper left', fontsize = 14)
plt.tight_layout()
plt.show()


threshold = 27.1
true = []
pred = []
true_pos = 0
pred_pos = 0
for x, y in test_ds:
    mean = x
    true.append(y.numpy())
    if mean < threshold:
        pred.append(0)
    if mean > threshold:
        pred.append(1)
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

fpr1 , tpr1 , thresholds = roc_curve(true, pred)

auc1 = metrics.auc(fpr1, tpr1)
plt.figure(2)
plot_roc_curve2(fpr1, tpr1, auc1)
print('AUC: ', auc1)
