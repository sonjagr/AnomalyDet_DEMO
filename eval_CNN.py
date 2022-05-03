import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from helpers.dataset_helpers import create_cnn_dataset
from autoencoders import *
from common import *
from scipy.ndimage.interpolation import rotate
import random, argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn import metrics
random.seed(42)
from sklearn.metrics import confusion_matrix, log_loss
tf.keras.backend.clear_session()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def plot_roc_curve(fpr1, tpr1, auc1, fpr2, tpr2, auc2):
    plt.plot(fpr1, tpr1, color = 'C1', label = 'CNN AUC = '+str(round(auc1, 2)))
    plt.plot(fpr2, tpr2, color = 'C0', label = 'AE baseline AUC = '+str(round(auc2, 2)))
    plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), linestyle = '--', color = 'gray')
    plt.xlabel('False Positive Rate/(1-Specificity)')
    plt.ylabel('True Positive Rate/Sensitivity')
    #plt.xscale('log')
    plt.grid()
    plt.legend(loc = 'lower right')
    plt.show()

def plot_examples(test_pred_plot, test_true_plot, test_img_plot, saveas):
    test_pred_plot = np.round(test_pred_plot, 2)
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))
    ax1.imshow(test_img_plot[0], vmin = 0, vmax = 255)
    ax1.set_title('Label: '+str(round(test_true_plot[0]))+'\nOutput: '+str(test_pred_plot[0]))
    ax1.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax2.imshow(test_img_plot[1], vmin = 0, vmax = 255)
    ax2.set_title('Label: '+str(round(test_true_plot[1]))+'\nOutput: '+str(test_pred_plot[1]))
    ax2.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax3.imshow(test_img_plot[2], vmin = 0, vmax = 255)
    ax3.set_title('Label: '+str(round(test_true_plot[2]))+'\nOutput: '+str(test_pred_plot[2]))
    ax3.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax4.imshow(test_img_plot[3], vmin = 0, vmax = 255)
    ax4.set_title('Label: '+str(round(test_true_plot[3]))+'\nOutput: '+str(test_pred_plot[3]))
    ax4.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax5.imshow(test_img_plot[4], vmin = 0, vmax = 255)
    ax5.set_title('Label: '+str(round(test_true_plot[4]))+'\nOutput: '+str(test_pred_plot[4]))
    ax5.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax6.imshow(test_img_plot[5], vmin = 0, vmax = 255)
    ax6.set_title('Label: '+str(round(test_true_plot[5]))+'\nOutput: '+str(test_pred_plot[5]))
    ax6.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    plt.tight_layout()
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/cnn_results_%s.png' % saveas, dpi=600)
    plt.show()

def rounding_thresh(input, thresh):
    rounded = np.round(input - thresh + 0.5)
    return rounded

savename = 'notf_noae'
savename = 'notf_withae_MTN4'
cont_epoch = 100

base_dir = 'db/'
dir_det = 'DET/'
MTN = 4

images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
val_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'val_img_list_%i.npy' % MTN)
val_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'val_lbl_list_%i.npy' % MTN)
test_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'test_img_list_%i.npy' % MTN)
test_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'test_lbl_list_%i.npy' % MTN)

#test_img_list = val_img_list
#test_lbl_list = val_lbl_list

test_loss = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/losses/%s/test_loss_%s.npy' % (savename, savename))
train_loss = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/losses/%s/train_loss_%s.npy' % (savename, savename))

test_pred_plot = [0,0,0,0,0,0]
test_true_plot = [0,0,0,0,0,0]
test_img_plot = test_img_list[12:18, :, :]
#plot_examples(test_pred_plot, test_true_plot, test_img_plot, '0')

'''
print(len(test_loss), len(train_loss))
plt.figure(1, figsize=(9, 6))
plt.plot(np.arange(1,len(test_loss)+1, len(test_loss)/len(train_loss)), train_loss, label = 'Training')
plt.plot(np.arange(1,len(test_loss)+1), test_loss, label = 'Validation')
plt.grid()
plt.xlim(0,140)
plt.legend(fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Binary cross-entropy', fontsize = 12)
plt.tight_layout()
plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/cnn_trainin.png', dpi = 600)
plt.show()
'''

test_scores = []

model = tf.keras.models.load_model('saved_class/%s/cnn_%s_epoch_%i' % (savename,savename,cont_epoch))
#model_noae =  tf.keras.models.load_model('saved_class/%s/cnn_%s_epoch_%i' % ('run2_noae','run2_noae',cont_epoch))
print(model.summary())

print('Testing data shape: ', len(test_img_list), len(test_lbl_list))

test_img_list = test_img_list.reshape(-1, 160, 160)
test_lbl_list = test_lbl_list.flatten()


test_pred = model.predict(test_img_list)
test_loss = log_loss(test_lbl_list, test_pred.astype("float64"))
print('test score', test_loss)

test_pred = model.predict(test_img_list).flatten()
fpr1 , tpr1 , thresholds = roc_curve(test_lbl_list, test_pred)
auc = metrics.auc(fpr1, tpr1)

fpr2 = np.load('fpr1.npy')
tpr2 = np.load('tpr1.npy')
auc2 = metrics.auc(fpr2, tpr2)


plot_roc_curve(fpr1, tpr1, auc, fpr2, tpr2, auc2)


print('AUC: ', auc)
test_pred_plot = test_pred[0:6]
test_true_plot = test_lbl_list[0:6]
test_img_plot = test_img_list[0:6, :, :]
#plot_examples(test_pred_plot, test_true_plot, test_img_plot, '0')
test_pred_plot = test_pred[6:12]
test_true_plot = test_lbl_list[6:12]
test_img_plot = test_img_list[6:12, :, :]
#plot_examples(test_pred_plot, test_true_plot, test_img_plot, '2')
test_pred_plot = test_pred[12:18]
test_true_plot = test_lbl_list[12:18]
test_img_plot = test_img_list[12:18, :, :]
#plot_examples(test_pred_plot, test_true_plot, test_img_plot, '1')

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.5)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.5).astype("float64"))
print('test score', test_loss)
print('TH 0.5: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.4)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.4).astype("float64"))
print('test score', test_loss)
print('TH 0.4: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.3)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.3).astype("float64"))
print('test score', test_loss)
print('TH 0.3: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.2)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.2).astype("float64"))
print('test score', test_loss)
print('TH 0.2: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.15)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.15).astype("float64"))
print('test score', test_loss)

print('TH 0.15: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.1)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.1).astype("float64"))
print('test score', test_loss)
print('TH 0.1: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.05)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.05).astype("float64"))
print('test score', test_loss)
print('TH 0.05: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))