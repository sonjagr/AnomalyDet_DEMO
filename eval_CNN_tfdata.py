import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from old_codes.autoencoders import *
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn import metrics
import tqdm
import seaborn as sns
random.seed(42)
from helpers.dataset_helpers import create_dataset
from helpers.cnn_helpers import ds_length, grad, preprocess
from sklearn.metrics import confusion_matrix, log_loss
tf.keras.backend.clear_session()
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def plot_roc_curve(fpr1, tpr1, auc1):
    plt.plot(fpr1, tpr1, color = 'C1', label = 'CNN, AUC = '+str(round(auc1, 2)))
    plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), linestyle = '--', color = 'gray')
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.ylabel('True Positive Rate', fontsize = 14)
    plt.scatter(0.1, 0.99, s=160, c='red', marker="*", label = 'Goal for ' +  r'$\bf{whole}$' + ' images', zorder = 5)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid()
    plt.legend(loc = 'lower right', fontsize = 14)
    plt.tight_layout()
    plt.show()

def plot_examples(test_pred_plot, test_true_plot, test_img_plot, saveas):
    test_pred_plot = np.round(test_pred_plot, 2)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 8))
    ax1.imshow(test_img_plot[0])
    ax1.set_title('Label: '+str(round(test_true_plot[0]))+'\nPrediction: '+str(test_pred_plot[0]), fontsize =16)
    ax1.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax2.imshow(test_img_plot[1])
    ax2.set_title('Label: '+str(round(test_true_plot[1]))+'\nPrediction: '+str(test_pred_plot[1]), fontsize =16)
    ax2.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax3.imshow(test_img_plot[2])
    ax3.set_title('Label: '+str(round(test_true_plot[2]))+'\nPrediction: '+str(test_pred_plot[2]), fontsize =16)
    ax3.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax4.imshow(test_img_plot[3])
    ax4.set_title('Label: '+str(round(test_true_plot[3]))+'\nPrediction: '+str(test_pred_plot[3]), fontsize =16)
    ax4.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    plt.tight_layout()
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/cnn_results_%s.png' % saveas, dpi=600)
    plt.show()

def rounding_thresh(input, thresh):
    rounded = np.round(input - thresh + 0.5)
    return rounded

def plot_losses(train_loss, val_loss):
    plt.figure(1, figsize=(8, 5))
    plt.plot(np.arange(1,len(val_loss)+1, len(val_loss)/len(train_loss)), train_loss, label = 'Training')
    plt.plot(np.arange(1,len(val_loss)+1), val_loss, label = 'Validation')
    plt.plot([160,160,160],[0,0.5 ,0.73], color = 'red', linestyle = '--', linewidth = 2)
    plt.grid()
    #plt.xlim(0,200)
    plt.ylim(0,np.max(val_loss))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize = 14)
    plt.xlabel('Epoch', fontsize = 14)
    plt.ylabel('Binary cross-entropy', fontsize = 14)
    plt.tight_layout()
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/cnn_trainin.png', dpi = 600)
    plt.show()

savename = 'testing_brightness_10'
epoch = 14
model_loc = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s_epoch_%i' % (savename, savename, epoch)
model = tf.keras.models.load_model(model_loc)

base_dir = 'db/'
dir_det = 'DET/'

ae = AutoEncoder()
print('Loading autoencoder and data...')
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')
base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
X_val_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)
X_val_det_list = [images_dir_loc + s for s in X_val_det_list]
Y_val_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()

N_det_val = int(len(X_val_det_list))
## how many samples used
X_val_det_list = X_val_det_list[int(N_det_val/2):]
Y_val_det_list = Y_val_det_list[int(N_det_val/2):]


N_det_val = int(len(X_val_det_list))
print('Loaded number of val samples: ', N_det_val)

val_loss = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s_val_loss.npy' % (savename, savename))
train_loss = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s_train_loss.npy' % (savename, savename))

val_imgs = create_dataset(X_val_det_list)
val_lbls = tf.data.Dataset.from_tensor_slices(Y_val_det_list)

batch_size = 1
val_combined_dataset_batch, val_ano_before, val_ano = preprocess(val_imgs, val_lbls, to_encode = True, ae = ae, normal_times = 50, to_augment = False, to_brightness = False, to_brightness_before_ae = False, batch_size = batch_size, drop_rem = False)
val_combined_dataset = val_combined_dataset_batch.unbatch()

print('Anomalous validation patches: ', val_ano_before, val_ano)

predictions = []
trues = []
for x, y in val_combined_dataset:
    x = tf.reshape(x, [-1, 160, 160, 1])
    y = tf.cast(y, tf.float32)
    pred = model(x, training = False).numpy()[0]
    predictions = np.append(predictions, pred)
    true = y.numpy()
    trues = np.append(trues, true)

fpr, tpr, thresholds = roc_curve(trues, predictions)
auc = metrics.auc(fpr, tpr)
print('AUC: ', auc)

plot_roc_curve(fpr, tpr, auc)

cm = confusion_matrix(trues, rounding_thresh(predictions, 0.01))
tn, fp, fn, tp = cm.ravel()
print('tn, fp, fn, tp: ', tn, fp, fn, tp)
error = (fp+fn)/(tp+tn+fp+fn)
print('Error: ', error)
acc = 1-error
print('Accuracy: ', acc)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

import sklearn
plt.figure(1)
cm_display = sklearn.metrics.ConfusionMatrixDisplay(cm).plot()
plt.show()

thresholds = np.arange(0.01, 0.5, 0.01)
fps = []
fns = []
for i in thresholds:
    tn, fp, fn, tp = confusion_matrix(trues, rounding_thresh(predictions, i)).ravel()
    fps.append(fp/(fp+tn))
    fns.append(fn/(fn+tp))

plt.title('Validation FPR and FNR')
plt.plot(thresholds, fps, label = 'FPR')
plt.plot(thresholds, fns, label = 'FNR')
plt.legend()
plt.grid()
plt.show()
