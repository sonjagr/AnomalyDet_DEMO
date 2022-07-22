import matplotlib.patches
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from helpers.dataset_helpers import create_cnn_dataset, box_index_to_coords, process_anomalous_df_to_numpy
from helpers.cnn_helpers import crop, encode
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import random, time
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix
import sklearn
tf.keras.backend.clear_session()

## USER SELECTS THESE
gpu = '3'
th = 0.03
plot_th = 1
history_is = 0
savename = 'clean_moredata_fl'
batch_size = 1
epoch = '20'

## USED FOR SETTING PATHS
database_dir = DataBaseFileLocation_gpu
base_dir = TrainDir_gpu
images_dir_loc = imgDir_gpu
dir_ae = 'AE/'
dir_det = 'DET/'

print('Analysing model: ' + savename + ', epoch: ' +str(epoch))

if gpu is not 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

random.seed(42)
np.random.seed(42)

def plot_training(df, savename):
    train_loss = df['loss']
    val_loss = df['val_loss']
    plt.plot(np.arange(0, len(train_loss)), train_loss, label = 'Training')
    plt.plot(np.arange(0, len(val_loss)), val_loss, label = 'Validation')
    #plt.ylim(0,1)
    #plt.xlim(150, 500)
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy Loss')
    plt.title('Training and validation losses as a function of epochs ' )
    plt.savefig(os.path.join(base_dir, 'saved_CNNs/%s/training_history_plot.png' % savename), dpi=600)
    plt.show()

if history_is:
    training_history_file = os.path.join(base_dir, 'saved_CNNs/%s/history_log.csv' % savename)
    history_df = pd.read_csv(training_history_file)
    plot_training(history_df, savename)

model = tf.keras.models.load_model(os.path.join(base_dir, 'saved_CNNs/%s/cnn_%s_epoch_%s' % (savename, savename, epoch)), compile=False)
br_model = tf.keras.models.load_model(os.path.join(base_dir, 'saved_CNNs/br_br_1_new/br_cnn_br_1_new'), compile=False)
print(model.summary())

ae = AutoEncoder()
print('    Loading autoencoder and data...')
ae.load(os.path.join(base_dir, 'checkpoints/TQ3_2_1_TQ3_2_more_params_2/AE_TQ3_2_277_to_277_epochs'))

###training data
f_train = os.path.join(base_dir, 'db/TRAIN_DATABASE_20220711')
with pd.HDFStore( f_train,  mode='r') as store:
        train_val_db = store.select('db')
        print(f'Reading {f_train}')
X_train_val_list, Y_train_val_list = process_anomalous_df_to_numpy(train_val_db)
_, X_val_det_list, _, Y_val_det_list = train_test_split(X_train_val_list, Y_train_val_list, test_size = 70, random_state = 42)
X_val_det_list = [images_dir_loc + s for s in X_val_det_list]
##

### testing data
f_test = os.path.join(base_dir, 'db/TEST_DATABASE_20220711')
with pd.HDFStore(f_test,  mode='r') as store:
        test_db = store.select('db')
        print(f'Reading {f_test}')
X_test_det_list, Y_test_det_list = process_anomalous_df_to_numpy(test_db)
X_test_det_list = [images_dir_loc + s for s in X_test_det_list]
Y_test_det_list = Y_test_det_list.tolist()
print(X_test_det_list[1])
N_det_test = len(Y_test_det_list)

X_test_norm_list_loaded = np.load(database_dir + 'NORMAL_TEST_20220711.npy', allow_pickle=True)
X_test_norm_list_loaded = [images_dir_loc + s for s in X_test_norm_list_loaded]

xtest = 5
X_test_norm_list = np.random.choice(X_test_norm_list_loaded, (int(N_det_test)*xtest)+74, replace=False)
Y_test_norm_list = np.full((len(X_test_norm_list), 408), 0.)
print('    Loaded number of ANOMALOUS test whole images: ', N_det_test)
print('    Number of AVAILABLE normal test whole images ', len(X_test_norm_list_loaded))
print('    Number of USED normal test whole images ', len(Y_test_norm_list))

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
    return image, label

@tf.function
def process_crop(image, label):
    image, label = crop(image, label)
    return image, label

def plot_cm(labels, predictions, p, label):
    cm = confusion_matrix(labels,predictions, normalize='true')
    import sklearn
    plt.figure(1)
    cm_display = sklearn.metrics.ConfusionMatrixDisplay(cm).plot()
    plt.title(label)
    plt.show()

def plot_roc(name, labels, predictions, title, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.figure(2)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,50])
    plt.ylim([50,100.5])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.savefig(os.path.join(base_dir, 'saved_CNNs/%s/roc_plot_epoch_%s.png' % (savename, epoch)), dpi=600)
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
    plt.savefig(os.path.join(base_dir, 'saved_CNNs/%s/prc_plot_epoch_%s.png' % (savename, epoch)), dpi=600)
    plt.show()

def rounding_thresh(input, thresh):
    rounded = np.round(input - thresh + 0.5)
    return rounded

def plot_histogram(labels, pred_flat, p, normalize, ylim):
    true_ano = []
    true_nor = []
    for i,j in zip(labels, pred_flat):
        if i == 0:
            true_nor.append(j)
        if i == 1:
            true_ano.append(j)
    print(len(true_nor), len(true_ano))
    plt.grid()
    plt.ylim(0, ylim)
    plt.hist(true_nor, bins = int(np.sqrt(len(true_nor))), color = 'green', density = normalize, alpha = 0.7, label = 'Normal')
    plt.hist(true_ano, bins = int(np.sqrt(len(true_ano))), color = 'red', density = normalize, alpha = 0.7, label = 'Anomalous')
    plt.legend()
    plt.show()

def plot_thresholds(val_labels, val_pred):
    thresholds = np.arange(0.001, 0.4, 0.005)
    val_pred_flat = np.array(val_pred).flatten()
    val_labels = np.array(val_labels).flatten()
    fps = []
    fns = []
    for thresh in thresholds:
        cm = confusion_matrix(val_labels, val_pred_flat > thresh, normalize='true')
        tn, fp, fn, tp = cm.ravel()
        fps.append(fp)
        fns.append(fn)
    plt.plot(thresholds, fps, label='FP')
    plt.plot(thresholds, fns, label='FN')
    plt.grid()
    plt.title('FP and FN for the validation set with different classification thresholds')
    plt.xlabel('Threshold')
    plt.ylabel('# of false predictions')
    plt.legend()
    plt.savefig(os.path.join(base_dir, 'saved_CNNs/%s/thresholds_plot_epoch_%s.png' % (savename, epoch)), dpi=600)
    plt.show()

def evaluate_preds(label, prediction, title):
    print(title)

    plot_cm(label, rounding_thresh(prediction, th), p = th, label='Confusion matrix for %s' % title)
    cm = confusion_matrix(label, rounding_thresh(prediction, th))

    if 'whole' not in title:
        plot_histogram(label, prediction ,th, True, 100)
    elif 'whole' in title:
        plot_histogram(label, prediction, th, False, 250)

    tn, fp, fn, tp = cm.ravel()
    print('tn, fp, fn, tp: ', tn, fp, fn, tp)

    print('Precision: ', np.round(tp / (tp + fp), 2))
    print('recall: ', np.round(tp / (tp + fn), 2))
    error = (fp + fn) / (tp + tn + fp + fn)
    print('Error: ', np.round(error, 2))

    acc = 1 - error
    print('Accuracy: ', np.round(acc, 2))

    f2 = sklearn.metrics.fbeta_score(label, rounding_thresh(prediction, th), beta = 2)
    print('Fbeta: ', np.round(f2, 2))

    print('FPR: ', np.round(fp / (fp + tn), 2))
    print('FNR: ', np.round(fn / (fn + tp), 2))

    if 'whole' not in title:
        plot_roc("Test", label, prediction, 'ROC curve for for %s' % title, linestyle='--')
        plot_prc("Test", label, prediction, 'PRC curve for for %s' % title, linestyle='--')

def plot_false(false_patches, title):
    l = int(len(false_patches)/(160*160))
    false_patches = false_patches.reshape(l, 160, 160)
    for i in range(0,l):
        plt.imshow(false_patches[i].reshape(160,160))
        plt.title(title)
        plt.show()

def plot_false_wholes(false_wholes, title):
    l = int(len(false_wholes) / (2720*3840))
    false_wholes = false_wholes.reshape(l, 2720, 3840)
    for i in range(0,l):
        false_whole = false_wholes[i].reshape(2720, 3840)
        plt.imshow(false_whole)
        plt.title(title)
        plt.show()

def format_data_batch(image, label):
    image = tf.reshape(image, [408, 160, 160, 1])
    label = tf.cast(label, tf.float32)
    return image, label

def eval_loop(dataset, test = 1, N=500):
    ind = 0
    whole_pred = []
    whole_label = []
    patch_pred = []
    patch_label = []
    plot = 1
    encode_times = []
    prediction_times = []
    false_positive_patches = []
    false_negative_patches = []
    false_negative_wholes = []
    for whole_img, whole_lbl in tqdm(dataset, total=N):
        ## crop and encode
        whole_img, whole_lbl = process_crop(whole_img, whole_lbl)
        enc_t1 = time.time()
        whole_img_enc, whole_lbl = encode(whole_img, whole_lbl, ae)
        enc_t2 = time.time()
        encode_times.append(enc_t2-enc_t1)

        ##patch
        whole_img_patched = patch_image(whole_img)
        whole_img_enc_patched = patch_image(whole_img_enc)
        whole_img_patched, whole_lbl = format_data_batch(whole_img_patched, whole_lbl)

        ##predict backgrounds
        br_pred = br_model.predict(whole_img_patched)
        br_pred = np.round(br_pred)
        br_pred_ids = np.where(br_pred > 0.5)[0]
        ## get backround as zeroes so acts as a filter
        br_pred_inv = 1-br_pred
        whole_img_enc_patched, whole_lbl = format_data_batch(whole_img_enc_patched, whole_lbl)

        ## predict labels
        pred_t1 = time.time()
        pred = model.predict(whole_img_enc_patched)
        pred_t2 = time.time()
        prediction_times.append(pred_t2-pred_t1)

        pred = np.multiply(pred, br_pred_inv)
        pred_ids = np.where(pred > th)[0]

        whole_lbl = whole_lbl.numpy().flatten()
        lbl_ids = np.where(whole_lbl == 1.)[0]

        patch_pred.append(pred.flatten())
        patch_label.append(whole_lbl)

        if len(pred_ids) > 0:
            whole_pred_i = 1
        elif len(pred_ids) == 0:
            whole_pred_i = 0
        whole_pred = np.append(whole_pred, whole_pred_i)
        if len(lbl_ids) > 0:
            whole_label_i = 1
        elif len(lbl_ids) == 0:
            whole_label_i = 0
        whole_label = np.append(whole_label, whole_label_i)
        if whole_label_i == 1 and whole_pred_i == 0:
            false_negative_wholes = np.append(false_negative_wholes, whole_img)
        if plot == 1 and test == 1:
            #print('Whole label : ', whole_label)
            img, ax = plt.subplots()
            ax.imshow(whole_img.numpy().reshape(2720, 3840))
            for t in lbl_ids:
                x_t,y_t = box_index_to_coords(t)
                plt.gca().add_patch(patches.Rectangle((x_t, y_t), BOXSIZE, BOXSIZE, linewidth=2, edgecolor='r', facecolor='none'))
                if t not in pred_ids:
                    false_negative_patches = np.append(false_negative_patches,whole_img_enc_patched[t].numpy().reshape(160, 160))
            for p in pred_ids:
                if p in lbl_ids:
                    color = 'g'
                elif p not in lbl_ids:
                    color = 'y'
                    false_positive_patches = np.append(false_positive_patches, whole_img_enc_patched[p].numpy().reshape(160,160))
                x_p, y_p = box_index_to_coords(p)
                plt.gca().add_patch(patches.Rectangle((x_p, y_p), BOXSIZE, BOXSIZE, linewidth=2, edgecolor=color, facecolor='none'))
            for b in br_pred_ids:
                x_b, y_b = box_index_to_coords(b)
                plt.gca().add_patch(
                    patches.Rectangle((x_b, y_b), BOXSIZE, BOXSIZE, linewidth=2, edgecolor='gray', facecolor='none'))
            plt.show()
        ind = ind + 1
        if ind > 10:
            plot = 0
    return whole_pred, whole_label, patch_pred, patch_label, false_positive_patches, false_negative_patches, false_negative_wholes

test_ds = create_cnn_dataset(np.append(X_test_det_list, X_test_norm_list, axis = 0), np.append(Y_test_det_list, Y_test_norm_list, axis = 0), _shuffle=False).shuffle(buffer_size=213).batch(1)

if plot_th == 1:
    val_det_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list.tolist(), _shuffle=False).batch(1)
    whole_pred_val, whole_label_val, patch_pred_val, patch_label_val, _, _, _ = eval_loop(val_det_ds, 0 , N= 70)
    plot_thresholds(val_labels = patch_label_val, val_pred = patch_pred_val)

print('CLASSIFICATION THRESHOLD: ', th)

whole_pred, whole_label, patch_pred, patch_label, false_positive_patches, false_negative_patches, false_negative_wholes = eval_loop(test_ds, 1, N= 500)

false_positive_patches = np.array(false_positive_patches)
false_negative_patches = np.array(false_negative_patches)
#plot_false(false_positive_patches, 'False POSITIVE')
plot_false(false_negative_patches, 'False NEGATIVE')

plot_false_wholes(false_negative_wholes, 'False NEGATIVE')

patch_label = np.array(patch_label).flatten()
patch_pred = np.array(patch_pred).flatten()
whole_label = np.array(whole_label).flatten()
whole_pred = np.array(whole_pred).flatten()

#print('Average encode time:', np.mean(encode_times))
#print('Average prediction time:', np.mean(prediction_times))

evaluate_preds(patch_label, patch_pred, 'test patches')
evaluate_preds(whole_label, whole_pred, 'test whole images')
