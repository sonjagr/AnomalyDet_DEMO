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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

tf.keras.backend.clear_session()

@tf.function
def compute_loss_test(model, x, y_ref):
    y = model(x, training = False)
    reconstruction_error = bce(y_ref, y)
    return reconstruction_error

savename = 'works3_bs128'
savename = 'notf'
#savename = 'works2_noae'
cont_epoch = 199

base_dir = 'db/'
dir_det = 'DET/'
MTN = 1

images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
test_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'test_img_list_bright_%i.npy' % MTN)
test_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'test_lbl_list_bright_%i.npy' % MTN)

test_loss = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/losses/%s/test_loss_%s.npy' % (savename, savename))
train_loss = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/losses/%s/train_loss_%s.npy' % (savename, savename))
print(len(test_loss), len(train_loss))
plt.figure(1, figsize=(9, 6))
plt.plot(np.arange(0,len(test_loss), len(test_loss)/len(train_loss)), train_loss, label = 'Training')
plt.plot(np.arange(0,len(test_loss)), test_loss, label = 'Validation')
plt.grid()
plt.legend(fontsize = 12)
plt.xlabel('Epoch', fontsize = 12)
plt.ylabel('Binary cross-entropy', fontsize = 12)
plt.tight_layout()
plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/cnn_trainin.png', dpi = 600)
plt.show()

test_scores = []

model = tf.keras.models.load_model('saved_class/%s/cnn_%s_epoch_%i' % (savename,savename,cont_epoch))
#model_noae =  tf.keras.models.load_model('saved_class/%s/cnn_%s_epoch_%i' % ('run2_noae','run2_noae',cont_epoch))
print(model.summary())

bce = tf.keras.losses.BinaryCrossentropy()

print('Testing data shape: ', len(test_img_list), len(test_lbl_list))

test_img_list = test_img_list.reshape(-1, 160, 160)
test_lbl_list = test_lbl_list.flatten()
processed_test_dataset = tf.data.Dataset.from_tensor_slices((test_img_list, test_lbl_list)).batch(1)

loss = tf.keras.metrics.Mean()
for test_x, y_ref in processed_test_dataset:
    loss(compute_loss_test(model, test_x, y_ref))
test_score = loss.result().numpy()

test_pred = model.predict(test_img_list).flatten()
fpr , tpr , thresholds = roc_curve(test_lbl_list, test_pred)

auc = metrics.auc(fpr, tpr)

def plot_roc_curve(fpr, tpr, auc):
    itpr = []
    for i in tpr:
        itpr.append(1-i)
    plt.plot(fpr, tpr, label = 'AUC = '+str(round(auc, 2)))
    #plt.axis([0, 1, 0, 1])
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
    ax1.imshow(test_img_plot[0])
    ax1.set_title('Label: '+str(round(test_true_plot[0]))+'\nOutput: '+str(test_pred_plot[0]))
    ax1.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax2.imshow(test_img_plot[1])
    ax2.set_title('Label: '+str(round(test_true_plot[1]))+'\nOutput: '+str(test_pred_plot[1]))
    ax2.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax3.imshow(test_img_plot[2])
    ax3.set_title('Label: '+str(round(test_true_plot[2]))+'\nOutput: '+str(test_pred_plot[2]))
    ax3.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax4.imshow(test_img_plot[3])
    ax4.set_title('Label: '+str(round(test_true_plot[3]))+'\nOutput: '+str(test_pred_plot[3]))
    ax4.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax5.imshow(test_img_plot[4])
    ax5.set_title('Label: '+str(round(test_true_plot[4]))+'\nOutput: '+str(test_pred_plot[4]))
    ax5.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax6.imshow(test_img_plot[5])
    ax6.set_title('Label: '+str(round(test_true_plot[5]))+'\nOutput: '+str(test_pred_plot[5]))
    ax6.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    plt.tight_layout()
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/cnn_results_%s.png' % saveas, dpi=600)
    plt.show()

plot_roc_curve(fpr, tpr, auc)
print('AUC: ', auc)
test_pred_plot = test_pred[0:6]
test_true_plot = test_lbl_list[0:6]
test_img_plot = test_img_list[0:6, :, :]
plot_examples(test_pred_plot, test_true_plot, test_img_plot, '0')
test_pred_plot = test_pred[6:12]
test_true_plot = test_lbl_list[6:12]
test_img_plot = test_img_list[6:12, :, :]
plot_examples(test_pred_plot, test_true_plot, test_img_plot, '2')
test_pred_plot = test_pred[12:18]
test_true_plot = test_lbl_list[12:18]
test_img_plot = test_img_list[12:18, :, :]
plot_examples(test_pred_plot, test_true_plot, test_img_plot, '1')
test_pred_plot = test_pred[-6::-1]
test_true_plot = test_lbl_list[-6::-1]
test_img_plot = test_img_list[-6::-1, :, :]
plot_examples(test_pred_plot, test_true_plot, test_img_plot, '3')
print('Testing score: ', test_score)

def rounding_thresh(input, thresh):
    rounded = np.round(input - thresh + 0.5)
    return rounded

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.5)).ravel()
print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.4)).ravel()
print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.3)).ravel()
print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.2)).ravel()
print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.15)).ravel()
print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.1)).ravel()
print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))