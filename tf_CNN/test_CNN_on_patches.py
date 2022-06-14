import matplotlib.patches
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from helpers.dataset_helpers import create_cnn_dataset
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import random, time
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix

tf.keras.backend.clear_session()

gpu = '2'
#savename = 'testing_model3'
savename = 'whole_patch_3'
batch_size = 512
epoch = 100

print('Analysing model: ' + savename + ', epoch: ' +str(epoch))

if gpu is not 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

random.seed(42)

ae = AutoEncoder()
print('Loading autoencoder and data...')
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_TQ3_more_data/AE_TQ3_318_to_318_epochs')

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_norm = 'AE/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET_very_cleaned.npy', allow_pickle=True)
X_test_norm_list = np.load(base_dir + dir_norm + 'X_test_AE.npy', allow_pickle=True)

X_test_det_list = [images_dir_loc + s for s in X_test_det_list]
X_test_norm_list = [images_dir_loc + s for s in X_test_norm_list]

Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET_very_cleaned.npy', allow_pickle=True).tolist()

N_det_test = int(len(Y_test_det_list)/2)

X_val_det_list = X_test_det_list[:N_det_test]
X_val_norm_list = X_test_norm_list[:N_det_test]
Y_val_det_list = Y_test_det_list[:N_det_test]

X_test_det_list = X_test_det_list[-N_det_test:]
X_test_norm_list = X_test_norm_list[-N_det_test:]
Y_test_det_list = Y_test_det_list[-N_det_test:]

N_det_val = len(X_val_det_list)

Y_test_norm_list = np.full((N_det_val, 408), 0.).tolist()
Y_val_norm_list = np.full((N_det_val, 408), 0.).tolist()

print('Loaded number of val, test samples: ', N_det_val, N_det_test)
print('All loaded. Starting processing...')
time1 = time.time()

test_ds = create_cnn_dataset(np.append(X_test_det_list, X_test_norm_list, axis = 0), np.append(Y_test_det_list,Y_test_norm_list, axis = 0), _shuffle=False)
val_ds = create_cnn_dataset(np.append(X_val_det_list, X_val_norm_list, axis = 0), np.append(Y_val_det_list,Y_val_norm_list, axis = 0), _shuffle=False)

model = tf.keras.models.load_model('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s_epoch_%s' % (savename, savename, epoch), compile=False)

def plot_metrics(history):
    plt.figure(2)
    colors = ['blue','red']
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig('saved_CNNs/%s/training.png' % savename, dpi = 600)
    plt.show()

def plot_examples(ds):
    for x, y in ds:
        plt.imshow(x)
        plt.title(str(y))
        plt.show()

@tf.function
def crop(img, lbl):
    img = tf.reshape(img, [-1, 2736, 3840, INPUT_DIM])
    img = tf.keras.layers.Cropping2D(cropping=((0, 16), (0, 0)))(img)
    return img, lbl

@tf.function
def encode(img, lbl, ae):
    img = tf.reshape(img, [-1, 2720, 3840, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.sqrt(tf.pow(tf.subtract(img, decoded_img), 2))
    return aed_img, lbl

@tf.function
def patch_images(img, lbl):
    split_img = tf.image.extract_patches(images=img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1], rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [17*24, 160 *160])
    lbl = tf.reshape(lbl, [17*24])
    patch_ds = tf.data.Dataset.from_tensors((re, lbl))
    return patch_ds

@tf.function
def patch_image(img):
    split_img = tf.image.extract_patches(images=img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1], rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [17*24, 160 *160])
    return re

def weighted_bincrossentropy(true, pred, weight_zero=1.0, weight_one=408.):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bin_crossentropy = bce(true, pred)

    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return tf.keras.backend.mean(weighted_bin_crossentropy)

@tf.function
def process_crop_encode(image, label):
    image, label = crop(image, label)
    image, label = encode(image, label, ae)
    return image,label

@tf.function
def process_crop(image, label):
    image, label = crop(image, label)
    return image,label

@tf.function
def format_imgs(image, label):
    image = tf.reshape(image, [160, 160, 1])
    label = tf.cast(label, tf.float32)
    return image, label

def plot_cm(labels, predictions, p):
    cm = confusion_matrix(labels, predictions>p, normalize='true')
    import sklearn
    plt.figure(1)
    cm_display = sklearn.metrics.ConfusionMatrixDisplay(cm).plot()
    plt.show()

import sklearn
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.figure(2)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,60])
    plt.ylim([40,100.5])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()

def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
    plt.figure(3)
    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()

def plot_training(df):
    train_loss = df['loss']
    val_loss = df['val_loss']
    plt.plot(np.arange(0, len(train_loss)), train_loss, label = 'Training')
    plt.plot(np.arange(0, len(val_loss)), val_loss, label = 'Validation')
    #plt.ylim(0,0.2)
    #plt.xlim(150, 500)
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy Loss')
    plt.show()

def rounding_thresh(input, thresh):
    rounded = np.round(input - thresh + 0.5)
    return rounded

test_ds_whole = test_ds
val_ds_whole = val_ds

test_ds = test_ds.map(process_crop_encode)
test_ds = test_ds.flat_map(patch_images).unbatch()
test_ds = test_ds.map(format_imgs)
test_ds_batch = test_ds.batch(408)

val_ds = val_ds.map(process_crop_encode)
val_ds = val_ds.flat_map(patch_images).unbatch()
val_ds = val_ds.map(format_imgs)
val_ds_batch = val_ds.batch(408)

test_labels = np.append(Y_test_det_list, Y_test_norm_list).flatten()
val_labels = np.append(Y_val_det_list, Y_val_norm_list).flatten()
print(len(Y_val_det_list), len(Y_val_norm_list))

##choose threshold for classification
th = 0.001
print('CLASSIFICATION THRESHOLD: ', th)
test_pred = model.predict(test_ds_batch)
test_pred_flat = test_pred.flatten()
val_pred = model.predict(val_ds_batch)

print(len(test_labels), len(test_pred))
print(len(val_labels), len(val_pred))

##load and plot training history
training_history_file = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/history_log.csv' % savename
history_df = pd.read_csv(training_history_file)
plot_training(history_df)
plt.show()

print('FOR PATCHES: ')
plot_cm(test_labels, test_pred_flat, p = th)

cm = confusion_matrix(test_labels, test_pred_flat > th, normalize='true')

tn, fp, fn, tp = cm.ravel()
print('tn, fp, fn, tp: ', tn, fp, fn, tp)

error = (fp+fn)/(tp+tn+fp+fn)
print('Error: ', error)

acc = 1-error
print('Accuracy: ', acc)

f2 = sklearn.metrics.fbeta_score(test_labels, test_pred_flat > th, beta = 2)
print('Fbeta: ', f2)

print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

plot_roc("Test", test_labels, test_pred_flat, linestyle='--')

plot_prc("Test", test_labels, test_pred_flat,  linestyle='--')

for name, metric in zip(model.metrics_names, model.metrics):
    print(name, ': ', metric(test_labels, test_pred_flat > th).numpy())
print()


