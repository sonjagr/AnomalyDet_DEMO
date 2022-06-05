import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from helpers.dataset_helpers import create_dataset
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import random, time
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

tf.keras.backend.clear_session()

gpu = '1'
savename = 'testing_modeltf_speed2'
batch_size = 1024
epoch = 900

if gpu is not 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

try:
    os.makedirs('saved_CNNs/%s' % savename)
except FileExistsError:
    pass

random.seed(42)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

ae = AutoEncoder()
print('Loading autoencoder and data...')
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)
X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_test_det_list = [images_dir_loc + s for s in X_test_det_list]
Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET.npy', allow_pickle=True).tolist()
Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()

N_det_test = int(len(X_test_det_list)/2)
X_train_det_list = X_train_det_list
Y_train_det_list = Y_train_det_list

X_test_det_list = X_test_det_list[-N_det_test:-10]
Y_test_det_list = Y_test_det_list[-N_det_test:-10]

N_det_train = len(X_train_det_list)
print('Loaded number of train, val samples: ', N_det_train, N_det_test)
print('All loaded. Starting processing...')
time1 = time.time()

train_imgs = create_dataset(X_train_det_list, _shuffle=False)
train_lbls = tf.data.Dataset.from_tensor_slices(Y_train_det_list)

test_imgs = create_dataset(X_test_det_list, _shuffle=False)
test_lbls = tf.data.Dataset.from_tensor_slices(Y_test_det_list)

test_ds = tf.data.Dataset.zip((test_imgs, test_lbls))

def plot_metrics(history):
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
def process_crop_encode(image, label):
    image, label = crop(image, label)
    image, label = encode(image, label, ae)
    return image,label

@tf.function
def format(image, label):
    image = tf.reshape(image, [160, 160, 1])
    label = tf.cast(label, tf.float32)
    return image, label

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    import sklearn
    plt.figure(1)
    cm_display = sklearn.metrics.ConfusionMatrixDisplay(cm).plot()
    plt.show()

import sklearn
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,60])
    plt.ylim([40,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

def plot_training(df):
    train_loss = df['loss']
    val_loss = df['val_loss']
    plt.plot(np.arange(0,len(train_loss)), train_loss, label = 'Training')
    plt.plot(np.arange(0,len(val_loss)), val_loss, label = 'Validation')
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy Loss')
    plt.show()

def rounding_thresh(input, thresh):
    rounded = np.round(input - thresh + 0.5)
    return rounded

test_ds = test_ds.map(process_crop_encode)
test_ds = test_ds.flat_map(patch_images).unbatch()
test_ds = test_ds.map(format)
test_ds_batch = test_ds.batch(batch_size)

test_labels = np.array(Y_test_det_list).flatten()

time2 = time.time()
pro_time = time2-time1
print('Processing (time {:.2f} s) finished, starting testing...'.format(pro_time))

model = tf.keras.models.load_model('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s_epoch_%s' % (savename, savename, epoch))

training_history_file = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/history_log.csv' % savename

history_df = pd.read_csv(training_history_file)

p = 0.9

plot_training(history_df)

test_pred = model.predict(test_ds_batch)
test_true = test_labels

plot_cm(test_true, test_pred, p = p)

cm = confusion_matrix(test_true, rounding_thresh(test_pred, p), normalize='true')
tn, fp, fn, tp = cm.ravel()
print('tn, fp, fn, tp: ', tn, fp, fn, tp)
error = (fp+fn)/(tp+tn+fp+fn)
print('Error: ', error)
acc = 1-error
print('Accuracy: ', acc)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

plot_roc("Test", test_true, test_pred, linestyle='--')
plt.legend(loc='lower right')
plt.show()

plot_prc("Test", test_true, test_pred,  linestyle='--')
plt.legend(loc='lower right')
plt.show()

for name, metric in zip(model.metrics_names, model.metrics):
    print(name, ': ', metric(test_true, rounding_thresh(test_pred, p)).numpy())
print()
