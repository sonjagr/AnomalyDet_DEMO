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

def plot_roc_curve(fpr1, tpr1, auc1, fpr2, tpr2, auc2):
    plt.plot(fpr1, tpr1, color = 'C1', label = 'CNN, AUC = '+str(round(auc1, 2)))
    plt.plot(fpr2, tpr2, color = 'C0', label = 'AE baseline, AUC = '+str(round(auc2, 2)))
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

savename = 'notf_noae'
savename = 'notf_withae_MTN4'
savename = 'larger_model4'
#savename = 'larger_model_10'
savename = 'larger_model4_whole2'
savename = 'larger_model4_weights'
cont_epoch = 15

base_dir = 'db/'
dir_det = 'DET/'
MTN = 4

images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
train_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'train_img_list_%i.npy' % MTN)
train_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'train_lbl_list_%i.npy' % MTN)
val_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'val_img_list_%i.npy' % MTN)
val_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'val_lbl_list_%i.npy' % MTN)
test_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'test_img_list_%i.npy' % MTN)
test_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'test_lbl_list_%i.npy' % MTN)

test_loss = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/losses/%s/test_loss_%s.npy' % (savename, savename))
train_loss = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/losses/%s/train_loss_%s.npy' % (savename, savename))

plt.figure(1, figsize=(8, 5))
plt.plot(np.arange(1,len(test_loss)+1, len(test_loss)/len(train_loss)), train_loss, label = 'Training')
plt.plot(np.arange(1,len(test_loss)+1), test_loss, label = 'Validation')
plt.plot([160,160,160],[0,0.5 ,0.73], color = 'red', linestyle = '--', linewidth = 2)
plt.grid()
#plt.xlim(0,200)
plt.ylim(0,np.max(test_loss))
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize = 14)
plt.xlabel('Epoch', fontsize = 14)
plt.ylabel('Binary cross-entropy', fontsize = 14)
plt.tight_layout()
plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/cnn_trainin.png', dpi = 600)
plt.show()

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
train_pred = model.predict(train_img_list).flatten()

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

val_pred = model.predict(val_img_list).flatten()

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

fpr1 , tpr1 , thresholds = roc_curve(test_lbl_list, test_pred)
auc = metrics.auc(fpr1, tpr1)

fpr2 = np.load('fpr1.npy')
tpr2 = np.load('tpr1.npy')
auc2 = metrics.auc(fpr2, tpr2)

plot_roc_curve(fpr1, tpr1, auc, fpr2, tpr2, auc2)

print('AUC: ', auc)
test_pred_plot = test_pred[1:5]
test_true_plot = test_lbl_list[1:5]
test_img_plot = test_img_list[1:5, :, :]
plot_examples(test_pred_plot, test_true_plot, test_img_plot, '0')

test_pred_plot = test_pred[10:14]
test_true_plot = test_lbl_list[10:14]
test_img_plot = test_img_list[10:14, :, :]
plot_examples(test_pred_plot, test_true_plot, test_img_plot, '0')

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.5)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.5).astype("float64"))
print('TH 0.5: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.4)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.4).astype("float64"))
print('TH 0.4: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.3)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.3).astype("float64"))
print('TH 0.3: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.2)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.2).astype("float64"))
print('TH 0.2: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.15)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.15).astype("float64"))
print('TH 0.15: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.1)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.1).astype("float64"))
print('TH 0.1: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.05)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.05).astype("float64"))
print('TH 0.05: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

tn, fp, fn, tp = confusion_matrix(test_lbl_list, rounding_thresh(test_pred, 0.02)).ravel()
test_loss = log_loss(test_lbl_list, rounding_thresh(test_pred, 0.02).astype("float64"))
print('TH 0.01: Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))