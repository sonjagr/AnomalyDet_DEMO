import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from old_codes.autoencoders import *
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn import metrics
random.seed(42)
from sklearn.metrics import confusion_matrix, log_loss
tf.keras.backend.clear_session()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    cmap = 'rainbow'
    ax1.imshow(test_img_plot[0], cmap = cmap)
    ax1.set_title('Label: '+str(round(test_true_plot[0]))+'\nPrediction: '+str(test_pred_plot[0]), fontsize =16)
    ax1.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax2.imshow(test_img_plot[1], cmap = cmap)
    ax2.set_title('Label: '+str(round(test_true_plot[1]))+'\nPrediction: '+str(test_pred_plot[1]), fontsize =16)
    ax2.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax3.imshow(test_img_plot[2], cmap = cmap)
    ax3.set_title('Label: '+str(round(test_true_plot[2]))+'\nPrediction: '+str(test_pred_plot[2]), fontsize =16)
    ax3.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    ax4.imshow(test_img_plot[3], cmap = cmap)
    ax4.set_title('Label: '+str(round(test_true_plot[3]))+'\nPrediction: '+str(test_pred_plot[3]), fontsize =16)
    ax4.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    plt.tight_layout()
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/cnn_results_%s.png' % saveas, dpi=600)
    plt.show()

def rounding_thresh(input, thresh):
    rounded = np.round(input - thresh + 0.5)
    return rounded

savename = 'more_normal_notf'
cont_epoch = 76

base_dir = 'db/'
dir_det = 'DET/'
MTN = 19

images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
train_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'train_img_list_aug_%i.npy' % MTN)
train_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'train_lbl_list_aug_%i.npy' % MTN)
val_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'val_img_list_%i.npy' % MTN)
val_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'val_lbl_list_%i.npy' % MTN)
test_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'test_img_list_%i.npy' % MTN)
test_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'test_lbl_list_%i.npy' % MTN)

test_loss = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s_test_loss.npy' % (savename, savename))
train_loss = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s_train_loss.npy' % (savename, savename))

model = tf.keras.models.load_model('saved_CNNs/%s/cnn_%s_epoch_%i' % (savename,savename,cont_epoch))

print('Testing data shape: ', len(test_img_list), len(test_lbl_list))

train_pred = model.predict(train_img_list).flatten()
val_pred = model.predict(val_img_list).flatten()

ano_pred = []
norm_pred = []
for i,j in zip(train_pred, train_lbl_list):
    if j == 0:
        norm_pred.append(i)
    if j == 1:
        ano_pred.append(i)
plt.hist(ano_pred, bins = int(np.sqrt(len(ano_pred))), density=True, color = 'red', alpha = 0.5, label = 'Anomalous')
plt.hist(norm_pred, bins = int(np.sqrt(len(norm_pred))), density=True,color = 'green',  alpha = 0.5, label = 'Non-anomalous')
plt.legend()
plt.grid()
plt.show()

predictions = model.predict(test_img_list).flatten()
trues = test_lbl_list

fpr, tpr, thresholds = roc_curve(trues, predictions)
auc = metrics.auc(fpr, tpr)
print('AUC: ', auc)

plot_roc_curve(fpr, tpr, auc)

cm = confusion_matrix(trues, rounding_thresh(predictions, 0.5), normalize='true')
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

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
plt.figure(2)
prec, recall, _ = precision_recall_curve(test_lbl_list, predictions, pos_label=1)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
plt.show()

thresholds = np.arange(0.01, 0.5, 0.01)
fps = []
fns = []
for i in thresholds:
    tn, fp, fn, tp = confusion_matrix(val_lbl_list, rounding_thresh(val_pred, i)).ravel()
    fps.append(fp/(fp+tn))
    fns.append(fn/(fn+tp))

plt.title('Validation FPR and FNR')
plt.plot(thresholds, fps, label = 'FPR')
plt.plot(thresholds, fns, label = 'FNR')
plt.legend()
plt.grid()
plt.show()