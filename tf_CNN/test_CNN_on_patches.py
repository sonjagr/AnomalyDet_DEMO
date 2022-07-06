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
from tqdm import tqdm
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix
import sklearn
tf.keras.backend.clear_session()

gpu = '3'
choose_test_ds = 'c'
th = 0.3
rgb = False
plot_ths = False
history_is = True
savename = 'final3_bayer_very_cleaned'
savename = 'cleaned_testing_vgg_2'
batch_size = 1
epoch = 36

database_dir = DataBaseFileLocation_gpu
base_dir = TrainDir_gpu
images_dir_loc = imgDir_gpu

print('Analysing model: ' + savename + ', epoch: ' +str(epoch))

if gpu is not 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

random.seed(42)

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
br_model = tf.keras.models.load_model(os.path.join(base_dir, 'saved_CNNs/br_br_testing_gputf/br_cnn_br_testing_gputf'), compile=False)

print(model.summary())

ae = AutoEncoder()
print('Loading autoencoder and data...')
ae.load(os.path.join(base_dir, 'checkpoints/TQ3_2_1_TQ3_2_more_params_2/AE_TQ3_2_193_to_193_epochs'))

dir_ae = 'AE/'
dir_det = 'DET/'

X_test_det_list = np.load(database_dir + dir_det + 'X_test_DET_final.npy', allow_pickle=True)

X_test_norm_list = np.load(database_dir + dir_ae + 'X_test_forcnn_clean.npy', allow_pickle=True)
X_test_norm_list = [s.replace('F:/ScratchDetection/MeasurementCampaigns/', '') for s in X_test_norm_list]

X_test_det_list = [images_dir_loc + s for s in X_test_det_list]
X_test_norm_list = [images_dir_loc + s for s in X_test_norm_list]

Y_test_det_list = np.load(database_dir + dir_det + 'Y_test_DET_final_cleaned.npy', allow_pickle=True).tolist()
Y_test_det_list = np.array(Y_test_det_list).reshape(int(len(Y_test_det_list)/408), 408)

N_det_test = int(len(Y_test_det_list)/2)

X_val_det_list = X_test_det_list[:N_det_test]
X_val_norm_list = X_test_norm_list[:N_det_test]
Y_val_det_list = Y_test_det_list[:N_det_test]

X_test_det_list = X_test_det_list[-N_det_test:]
X_test_norm_list = X_test_norm_list[-N_det_test:]
Y_test_det_list = Y_test_det_list[-N_det_test:]

Y_test_norm_list = np.full((N_det_test, 408), 0.).tolist()
Y_val_norm_list = np.full((N_det_test, 408), 0.).tolist()

print('Loaded number of val, test samples: ', N_det_test)
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
    return image,label

@tf.function
def process_crop_encode_rgb(image, label):
    image, label = crop(image, label)
    image, label = encode_rgb(image, label, ae)
    return image,label

def plot_cm(labels, predictions, p, label):
    cm = confusion_matrix(labels,predictions, normalize='true')
    import sklearn
    plt.figure(1)
    cm_display = sklearn.metrics.ConfusionMatrixDisplay(cm).plot()
    plt.title(label)
    plt.show()

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

def plot_histogram(labels, pred_flat, p):
    true_ano = []
    true_nor = []
    for i,j in zip(labels, pred_flat):
        if i == 0:
            true_nor.append(j)
        if i == 1:
            true_ano.append(j)
    plt.hist(true_nor, bins = int(np.sqrt(len(true_nor))), color = 'green', density = True, alpha = 0.7, label = 'Normal')
    plt.hist(true_ano, bins = int(np.sqrt(len(true_ano))), color = 'red', density = True, alpha = 0.7, label = 'Anomalous')
    plt.grid()
    plt.legend()
    plt.show()

def plot_thresholds():
    thresholds = np.arange(0.001, 0.2, 0.001)
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

    plot_histogram(label, prediction ,th)

    tn, fp, fn, tp = cm.ravel()
    print('tn, fp, fn, tp: ', tn, fp, fn, tp)

    print('Precision: ', np.round(tp / (tp + fp),2))
    print('recall: ', np.round(tp / (tp + fn),2))
    error = (fp + fn) / (tp + tn + fp + fn)
    print('Error: ', np.round(error, 2))

    acc = 1 - error
    print('Accuracy: ', np.round(acc, 2))

    f2 = sklearn.metrics.fbeta_score(label, rounding_thresh(prediction, th), beta=2)
    print('Fbeta: ', np.round(f2,2))

    print('FPR: ', np.round(fp / (fp + tn), 2))
    print('FNR: ', np.round(fn / (fn + tp), 2))

    plot_roc("Test", label, prediction, 'ROC curve for for %s' % title, linestyle='--')
    plot_prc("Test", label, prediction, 'PRC curve for for %s' % title, linestyle='--')

def plot_false(false_patches, title):
    false_patches = false_patches
    l = int(len(false_patches)/(160*160))
    false_patches = false_patches.reshape(l, 160, 160)
    for i in range(0,l):
        plt.imshow(false_patches[i].reshape(160,160))
        plt.title(title)
        plt.show()

def format_data_batch(image, label):
    image = tf.reshape(image, [408, 160, 160, 1])
    label = tf.cast(label, tf.float32)
    return image, label

def eval_loop(dataset):
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
    for whole_img, whole_lbl in tqdm(dataset, total=200):
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
            whole_pred.append(1)
        elif len(pred_ids) == 0:
            whole_pred.append(0)
        if len(lbl_ids) > 0:
            whole_label.append(1)
        elif len(lbl_ids) == 0:
            whole_label.append(0)
        if plot == 1:
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
    return whole_pred, whole_label, patch_pred , patch_label, false_positive_patches , false_negative_patches

test_ds = create_cnn_dataset(np.append(X_test_det_list, X_test_norm_list, axis = 0), np.append(Y_test_det_list, Y_test_norm_list, axis = 0), _shuffle=False).batch(1)
val_ds = create_cnn_dataset(np.append(X_val_det_list, X_val_norm_list, axis = 0), np.append(Y_val_det_list, Y_val_norm_list, axis = 0), _shuffle=False).batch(1)
print(len(list(val_ds)))
normal_test_ds = create_cnn_dataset(X_test_norm_list, Y_test_norm_list, _shuffle=False)
anomalous_test_ds = create_cnn_dataset(X_test_det_list, Y_test_det_list, _shuffle=False)

if choose_test_ds == 'a':
    test_ds = anomalous_test_ds
elif choose_test_ds == 'n':
    test_ds = normal_test_ds

test_ds = test_ds

##choose threshold for classification
print('CLASSIFICATION THRESHOLD: ', th)

whole_pred, whole_label, patch_pred , patch_label, false_positive_patches , false_negative_patches = eval_loop(test_ds)

false_positive_patches = np.array(false_positive_patches)
false_negative_patches = np.array(false_negative_patches)
#plot_false(false_positive_patches, 'False POSITIVE')
plot_false(false_negative_patches, 'False NEGATIVE')

patch_label = np.array(patch_label).flatten()
patch_pred = np.array(patch_pred).flatten()
whole_label = np.array(whole_label).flatten()
whole_pred = np.array(whole_pred).flatten()

#print('Average encode time:', np.mean(encode_times))
#print('Average prediction time:', np.mean(prediction_times))

evaluate_preds(patch_label, patch_pred, 'test patches')
evaluate_preds(whole_label, whole_pred, 'test whole images')



