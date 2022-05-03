import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from helpers.dataset_helpers import create_cnn_dataset
from autoencoders import *
from common import *
from sklearn.metrics import confusion_matrix, log_loss
from scipy.ndimage.interpolation import rotate
import random, argparse
import matplotlib.pyplot as plt
random.seed(42)

gpu = "3"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

bce = tf.keras.losses.BinaryCrossentropy()

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

#how many normal images for each defective
MTN = 1
train_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'train_img_list_noaug_%i.npy' % MTN)
train_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'train_lbl_list_noaug_%i.npy' % MTN)
test_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'val_img_list_%i.npy' % MTN)
test_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'val_lbl_list_%i.npy' % MTN)

train_means = []
for x in train_img_list:
    train_means.append(np.mean(x))

from sklearn.preprocessing import MinMaxScaler
train_means = np.array(train_means).reshape(-1, 1)
scaler = MinMaxScaler()
train_means = scaler.fit_transform(train_means).flatten()

test_means = []
for x in test_img_list:
    test_means.append(np.mean(x))

test_means = np.array(test_means).reshape(-1, 1)
test_means = scaler.transform(test_means).flatten()

print(len(train_img_list), len(test_img_list))
processed_test_dataset = tf.data.Dataset.from_tensor_slices((test_img_list, test_lbl_list))

normal, anomalous = [], []
normal_test, anomalous_test = [],[]
for x, y in zip(train_means, train_lbl_list):
    if y == 0:
        normal.append(x)
    if y == 1:
        anomalous.append(x)

th = 0.54
plt.figure(1)
plt.hist(anomalous,  bins = int(np.sqrt(len(anomalous))), density = True, alpha = 0.5, color = 'red', label='Anomalous', zorder = 3)
plt.hist(normal, bins = int(np.sqrt(len(normal))),  density = True, alpha = 0.5,color = 'green', label = 'Non-anomalous',zorder = 3)
plt.grid(zorder = 1)
plt.xlabel('Mean pixel-wise reconstruction error', fontsize = 12)
#plt.plot([th, th,th,th], [0.0,0.05,0.2,0.45], linestyle = '--', color = 'black', label='Threshold = 27.1')
plt.plot([th, th,th,th], [0.0,4,6,8.6], linestyle = '--', color = 'black', label='Threshold = 0.54')
plt.ylim(0,8.5)
plt.legend(loc = 'upper left', fontsize = 12)
plt.show()

threshold = 27.1
true = []
pred = []
true_pos = 0
pred_pos = 0
for x, y in tqdm(processed_test_dataset, total=tf.data.experimental.cardinality(processed_test_dataset).numpy()):
    true.append(y)
    if y == 0:
        normal_test.append(np.mean(x))
    if y == 1:
        true_pos = true_pos +1
        anomalous_test.append(np.mean(x))
    mean = np.mean(x)
    if mean < threshold:
        pred.append(0)
    if mean > threshold:
        pred.append(1)
        pred_pos = pred_pos + 1
print(true_pos)
print(pred_pos)

def rounding_thresh(input, thresh):
    rounded = np.round(input - thresh + 0.5)
    return rounded

pred = rounding_thresh(np.array(test_means), 0.53)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

print('LOGLOSS: ', bce(true, pred.astype("float64")))
tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0,1]).ravel()

cm= confusion_matrix(true, pred, labels=[0,1])
print(cm)
print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))

def plot_roc_curve2(fpr1, tpr1, auc1, ):
    plt.plot(fpr1, tpr1, color = 'C0', label = 'AE baseline AUC = '+str(round(auc1, 2)))
    plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), linestyle = '--', color = 'gray')
    plt.xlabel('False Positive Rate/(1-Specificity)', fontsize = 12)
    plt.ylabel('True Positive Rate/Sensitivity', fontsize = 12)
    #plt.xscale('log')
    plt.grid()
    plt.legend(loc = 'lower right', fontsize = 12)
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/baseline.png', dpi=600)
    plt.show()
from sklearn.metrics import roc_curve
from sklearn import metrics

fpr1 , tpr1 , thresholds = roc_curve(true, test_means)

auc1 = metrics.auc(fpr1, tpr1)
plt.figure(2)
plot_roc_curve2(fpr1, tpr1, auc1)
print('AUC: ', auc1)