import matplotlib.patches
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from helpers.dataset_helpers import create_cnn_dataset, box_index_to_coords, box_to_labels
from helpers.cnn_helpers import plot_examples, plot_metrics, crop, bayer2rgb, tf_bayer2rgb, encode, encode_rgb, format_data,patch_images
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import random, time
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix

tf.keras.backend.clear_session()

gpu = '2'
choose_test_ds = 'a'
th = 0.6
#savename = 'testing_model3'
savename = 'whole_smaller_patch_rgb'
batch_size = 1
epoch = 120

print('Analysing model: ' + savename + ', epoch: ' +str(epoch))

if gpu is not 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

random.seed(42)

training_history_file = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/history_log.csv' % savename
history_df = pd.read_csv(training_history_file)

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

model = tf.keras.models.load_model('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s_epoch_%s' % (savename, savename, epoch), compile=False)

@tf.function
def patch_image(img):
    split_img = tf.image.extract_patches(images=img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1], rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [17*24, 160 *160])
    return re

@tf.function
def process_crop_encode(image, label):
    image, label = crop(image, label)
    image, label = encode(image, label, ae)
    return image,label

@tf.function
def process_crop_encode_rgb(image, label):
    image, label = crop(image, label)
    image, label = encode_rgb(image, label, ae)
    return image,label

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
    plt.ylim(0,1.5)
    #plt.xlim(150, 500)
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy Loss')
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/training_history_plot.png' % savename, dpi=600)
    plt.show()

def rounding_thresh(input, thresh):
    rounded = np.round(input - thresh + 0.5)
    return rounded

test_ds = create_cnn_dataset(np.append(X_test_det_list, X_test_norm_list, axis = 0), np.append(Y_test_det_list,Y_test_norm_list, axis = 0), _shuffle=False)
val_ds = create_cnn_dataset(np.append(X_val_det_list, X_val_norm_list, axis = 0), np.append(Y_val_det_list,Y_val_norm_list, axis = 0), _shuffle=False)

normal_test_ds = create_cnn_dataset( X_test_norm_list, Y_test_norm_list, _shuffle=False)
anomalous_test_ds = create_cnn_dataset(X_test_det_list, Y_test_det_list, _shuffle=False)

test_ds_whole = test_ds
val_ds_whole = val_ds

if choose_test_ds == 'a':
    test_labels = np.array(Y_test_det_list).flatten()
    test_ds = anomalous_test_ds
if choose_test_ds == 'n':
    test_labels = np.array(Y_test_norm_list).flatten()
    test_ds = normal_test_ds
if choose_test_ds == 'c':
    test_labels = np.append(Y_test_det_list, Y_test_norm_list).flatten()

val_labels = np.append(Y_val_det_list, Y_val_norm_list).flatten()

print(len(Y_val_det_list), len(Y_val_norm_list))

test_ds = test_ds.map(process_crop_encode_rgb)
test_ds = test_ds.flat_map(patch_images).unbatch()
test_ds = test_ds.map(format_data)
test_ds_batch = test_ds.batch(408)

val_ds = val_ds.map(process_crop_encode_rgb)
val_ds = val_ds.flat_map(patch_images).unbatch()
val_ds = val_ds.map(format_data)
val_ds_batch = val_ds.batch(408)

test_pred = model.predict(test_ds_batch)
test_pred_flat = test_pred.flatten()
val_pred = model.predict(val_ds_batch)
val_pred_flat = val_pred.flatten()

thresholds = np.arange(0.001, 0.2, 0.001)
fps = []
fns = []
for thresh in thresholds:
    cm = confusion_matrix(val_labels, val_pred_flat > thresh, normalize='true')
    tn, fp, fn, tp = cm.ravel()
    fps.append(fp)
    fns.append(fn)
plt.plot(thresholds, fps, label = 'FP')
plt.plot(thresholds, fns, label = 'FN')
plt.grid()
plt.title('FP and FN for the validation set with different classification thresholds')
plt.xlabel('Threshold')
plt.ylabel('# of false predictions')
plt.legend()
plt.show()

##choose threshold for classification
print('CLASSIFICATION THRESHOLD: ', th)

ind = 0
normal = 0
defective = 0
##plot some examples:
for x, y in test_ds_whole.take(5):
    fig, ax = plt.subplots()
    x_patch, y_patch = crop(x, y)
    whole_y = y_patch.numpy().flatten()
    inds = np.where(whole_y == 1.)[0]
    inds = np.array(inds).astype('int')
    whole_img = x_patch.numpy().reshape(2720, 3840)
    ax.imshow(whole_img)
    normal = normal + 1
    if len(inds) > 0:
        for i in inds:
            true_x, true_y = box_index_to_coords(i)
            rec = matplotlib.patches.Rectangle((true_x, true_y), 160, 160, facecolor='None', edgecolor='red')
            ax.add_patch(rec)
    predictions = test_pred_flat[ind*408:(ind+1)*408]
    predictions = np.array(predictions).flatten()
    indspred = np.where(predictions > th)[0]
    indspred = np.array(indspred).astype('int')
    print(indspred)
    if len(indspred) > 0:
        defective = defective + 1
        for j in indspred:
            true_x, true_y = box_index_to_coords(j)
            if j not in inds:
                rec = matplotlib.patches.Rectangle((true_x, true_y), 160, 160, facecolor='None', edgecolor='yellow')
            if j in inds:
                rec = matplotlib.patches.Rectangle((true_x, true_y), 160, 160, facecolor='None', edgecolor='green')
            ax.add_patch(rec)
    if len(indspred) == 0:
        normal = normal + 1
    ind = ind + 1
    plt.show()

print('Number of flagged as anomalous, normals, all: ', defective, normal, ind)

##load and plot training history
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


