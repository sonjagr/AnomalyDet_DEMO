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

gpu = '5'
choose_test_ds = 'c'
th = 0.05
rgb = True
data = 'old'
plot_ths = True
history_is = False
savename = 'final3_bayer_very_cleaned'
savename = 'whole_batch_rgb_4'
batch_size = 1
epoch = 90

print('Analysing model: ' + savename + ', epoch: ' +str(epoch))

if gpu is not 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

random.seed(42)

def plot_training(df, savename):
    train_loss = df['loss']
    val_loss = df['val_loss']
    plt.plot(np.arange(0, len(train_loss)), train_loss, label = 'Training')
    plt.plot(np.arange(0, len(val_loss)), val_loss, label = 'Validation')
    plt.ylim(0,1)
    #plt.xlim(150, 500)
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy Loss')
    plt.title('Training and validation losses as a function of epochs of %s' % savename)
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/training_history_plot.png' % savename, dpi=600)
    plt.show()

if history_is:
    training_history_file = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/history_log.csv' % savename
    history_df = pd.read_csv(training_history_file)
    plot_training(history_df, savename)

#model = tf.keras.models.load_model('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s_epoch_%s' % (savename, savename, epoch), compile=False)
model = tf.keras.models.load_model('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s' % (savename, savename), compile=False)

print(model.summary())

ae = AutoEncoder()
print('Loading autoencoder and data...')
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_TQ3_more_data/AE_TQ3_318_to_318_epochs')

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_norm = 'AE/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

if data=='old':
    X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)
if data == 'new':
    X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET_2.npy', allow_pickle=True)
if data == 'very_clean':
    X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET_very_cleaned.npy', allow_pickle=True)

X_test_norm_list = np.load(base_dir + dir_norm + 'X_test_AE.npy', allow_pickle=True)

X_test_det_list = [images_dir_loc + s for s in X_test_det_list]
X_test_norm_list = [images_dir_loc + s for s in X_test_norm_list]

if data == 'old':
    Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()
if data == 'new':
    Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET_2.npy', allow_pickle=True).tolist()
if data == 'very_clean':
    Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET_very_cleaned.npy', allow_pickle=True).tolist()

N_det_test = int(len(Y_test_det_list)/2)

X_val_det_list = X_test_det_list[:N_det_test]
X_val_norm_list = X_test_norm_list[:N_det_test]
Y_val_det_list = Y_test_det_list[:N_det_test]

X_test_det_list = X_test_det_list[-N_det_test:]
X_test_norm_list = X_test_norm_list[-N_det_test:]
Y_test_det_list = Y_test_det_list[-N_det_test:]

N_det_val = len(X_val_det_list)

if data == 'very_clean':
    Y_test_norm_list = np.full((N_det_test, 408), 0.).tolist()
    Y_val_norm_list = np.full((N_det_val, 408), 0.).tolist()
else:
    Y_test_norm_list = np.full((N_det_test, 17, 24), 0.).tolist()
    Y_val_norm_list = np.full((N_det_val, 17, 24), 0.).tolist()

print('Loaded number of val, test samples: ', N_det_val, N_det_test)
print('All loaded. Starting processing...')
time1 = time.time()

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

def plot_cm(labels, predictions, p, label):
    cm = confusion_matrix(labels, predictions>p, normalize='true')
    import sklearn
    plt.figure(1)
    cm_display = sklearn.metrics.ConfusionMatrixDisplay(cm).plot()
    plt.title(label)
    plt.show()

import sklearn
def plot_roc(name, labels, predictions, title, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.figure(2)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,60])
    plt.ylim([40,100.5])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/roc_plot_epoch_%s.png' % (savename, epoch), dpi=600)
    plt.show()

def plot_prc(name, labels, predictions, title, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
    plt.figure(3)
    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/prc_plot_epoch_%s.png' % (savename, epoch), dpi=600)
    plt.show()

def rounding_thresh(input, thresh):
    rounded = np.round(input - thresh + 0.5)
    return rounded

def plot_histogram(test_labels, test_pred_flat):
    true_ano = []
    true_nor = []
    for i,j in zip(test_labels, test_pred_flat):
        if i == 0:
            true_nor.append(j)
        if i == 1:
            true_ano.append(j)
    plt.hist(true_nor, color = 'green')
    plt.hist(true_ano, color = 'red')
    plt.show()

def plot_wrong_test_examples(test_labels, test_pred_flat, test_ds):
    for i in range(0, len(test_labels)):
        if test_labels[i] != test_pred_flat:
            plt.imshow(test_ds[i])

test_ds = create_cnn_dataset(np.append(X_test_det_list, X_test_norm_list, axis = 0), np.append(Y_test_det_list, Y_test_norm_list, axis = 0), _shuffle=False)
val_ds = create_cnn_dataset(np.append(X_val_det_list, X_val_norm_list, axis = 0), np.append(Y_val_det_list, Y_val_norm_list, axis = 0), _shuffle=False)

normal_test_ds = create_cnn_dataset( X_test_norm_list, Y_test_norm_list, _shuffle=False)
anomalous_test_ds = create_cnn_dataset(X_test_det_list, Y_test_det_list, _shuffle=False)

if choose_test_ds == 'a':
    test_labels = np.array(Y_test_det_list).flatten()
    test_ds = anomalous_test_ds
if choose_test_ds == 'n':
    test_labels = np.array(Y_test_norm_list).flatten()
    test_ds = normal_test_ds
if choose_test_ds == 'c':
    test_labels = np.append(Y_test_det_list, Y_test_norm_list, axis = 0).flatten()

test_ds_whole = test_ds
val_ds_whole = val_ds

val_labels = np.append(Y_val_det_list, Y_val_norm_list, axis = 0).flatten()

if rgb == True:
    test_ds = test_ds.map(process_crop_encode_rgb)
    val_ds = val_ds.map(process_crop_encode_rgb)
if rgb == False:
    time1 = time.time()
    test_ds = test_ds.map(process_crop_encode)
    time2 = time.time()
    val_ds = val_ds.map(process_crop_encode)


test_ds = test_ds.flat_map(patch_images).unbatch()
test_ds = test_ds.map(format_data)
test_ds_batch = test_ds.batch(408)

val_ds = val_ds.flat_map(patch_images).unbatch()
val_ds = val_ds.map(format_data)
val_ds_batch = val_ds.batch(408)

def clean(test_pred, test_ds, p):
    test_pred = test_pred.flatten()
    i = 0
    for x, y in test_ds:
        if test_pred[i] > p:
            patch = x.numpy().flatten()
            maximum = np.max(patch)*0.8
            za = (patch > maximum).sum()
            print(za)
            if za < 10:
                print('label cleaned')
                test_pred[i] = 0
        i = i+1
    return test_pred

time_1 = time.time()
test_pred = model.predict(test_ds_batch)

#test_pred = clean(test_pred, test_ds, th)
time_2 = time.time()
print('Prediction time [s] for whole images (408 patches): ', (time_2 - time_1)/N_det_test)

test_pred_flat = test_pred.flatten()
val_pred = model.predict(val_ds_batch)
val_pred_flat = val_pred.flatten()

if plot_ths == True:
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
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/thresholds_plot_epoch_%s.png' % (savename, epoch), dpi=600)
    plt.show()

##choose threshold for classification
print('CLASSIFICATION THRESHOLD: ', th)

ind = 0
normal = 0
defective = 0
whole_pred = []
whole_label = []
plot = 1
encode_times = []
patch_times = []
for x, y in test_ds_whole:
    if plot == 1:
        fig, ax = plt.subplots()
    x_patch, y_patch = crop(x, y)
    enct1 = time.time()
    _, _ = encode(x_patch, y_patch, ae)
    enct2 = time.time()
    encode_times.append(enct2-enct1)
    whole_y = y_patch.numpy().flatten()
    inds = np.where(whole_y == 1.)[0]
    inds = np.array(inds).astype('int')
    whole_img = x_patch.numpy().reshape(2720, 3840)
    if plot == 1:
        ax.imshow(whole_img)
    if len(inds) > 0:
        whole_label.append(1.)
        if plot == 1:
            for i in inds:
                true_x, true_y = box_index_to_coords(i)
                rec = matplotlib.patches.Rectangle((true_x, true_y), 160, 160, facecolor='None', edgecolor='red')
                ax.add_patch(rec)
    if len(inds) == 0:
        whole_label.append(0.)
    predictions = test_pred_flat[ind*408:(ind+1)*408]
    predictions = np.array(predictions).flatten()
    indspred = np.where(predictions > th)[0]
    indspred = np.array(indspred).astype('int')
    if len(indspred) > 0:
        whole_pred.append(1.)
        defective = defective + 1
        if plot == 1:
            for j in indspred:
                true_x, true_y = box_index_to_coords(j)
                if j not in inds:
                    rec = matplotlib.patches.Rectangle((true_x, true_y), 160, 160, facecolor='None', edgecolor='yellow')
                if j in inds:
                    rec = matplotlib.patches.Rectangle((true_x, true_y), 160, 160, facecolor='None', edgecolor='green')
                ax.add_patch(rec)
    if len(indspred) == 0:
        whole_pred.append(0.)
        normal = normal + 1
    ind = ind + 1
    if ind > 10 and ind < 20:
        plot = 0
    if plot == 1:
        plt.show()

print('Number of whole images flagged as anomalous, normals, all: ', defective, normal, ind)
print('Average encode time:', np.mean(encode_times))
print('FOR PATCHES: ')
plot_cm(test_labels, test_pred_flat, p = th, label = 'Confusion matrix for test patches')
cm = confusion_matrix(test_labels, test_pred_flat > th, normalize='true')

plot_histogram(test_labels, test_pred_flat)

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

plot_roc("Test", test_labels, test_pred_flat, 'ROC curve for test patches', linestyle='--')
plot_prc("Test", test_labels, test_pred_flat, 'PRC curve for test patches',  linestyle='--')

for name, metric in zip(model.metrics_names, model.metrics):
    print(name, ': ', metric(test_labels, test_pred_flat > th).numpy())
print()

whole_pred = np.array(whole_pred)
whole_label = np.array(whole_label)
print('FOR WHOLE IMAGES')
plot_cm(whole_label, whole_pred, p = 0.5, label = 'Confusion matrix for test whole images')
cm = confusion_matrix(whole_label, whole_pred, normalize='true')

tn, fp, fn, tp = cm.ravel()
print('tn, fp, fn, tp: ', tn, fp, fn, tp)

error = (fp+fn)/(tp+tn+fp+fn)
print('Error: ', error)

acc = 1-error
print('Accuracy: ', acc)

f2 = sklearn.metrics.fbeta_score(whole_label, whole_pred, beta = 2)
print('Fbeta: ', f2)

print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

