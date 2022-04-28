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
random.seed(42)

gpu = "4"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

def classifier_scores(test_loss, train_loss, title):
    plt.plot(np.arange(0,len(test_loss)), test_loss, label = 'Test loss')
    plt.plot(np.arange(0,len(train_loss)), train_loss, linestyle= '--', label = 'Train loss')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Binary cross-entropy')
    plt.title(title)
    plt.show()

bce = tf.keras.losses.BinaryCrossentropy()

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

#how many normal images for each defective
MTN = 1
train_img_list = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'train_img_list_noaug_%i.npy' % MTN)
train_lbl_list = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'train_lbl_list_noaug_%i.npy' % MTN)
test_img_list = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'test_img_list_%i.npy' % MTN)
test_lbl_list = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'test_lbl_list_%i.npy' % MTN)

processed_train_dataset = tf.data.Dataset.from_tensor_slices((train_img_list, train_lbl_list))
processed_test_dataset = tf.data.Dataset.from_tensor_slices((test_img_list, test_lbl_list))

normal, anomalous = [], []
normal_test, anomalous_test = [],[]
for x, y in tqdm(processed_train_dataset, total=tf.data.experimental.cardinality(processed_train_dataset).numpy()):
    if y == 0:
        normal.append(np.mean(x))
    if y == 1:
        anomalous.append(np.mean(x))
print(len(normal), len(anomalous))

plt.hist(anomalous,  bins = int(np.sqrt(len(anomalous))), density = True, alpha = 0.5, color = 'red', label='Anomalous', zorder = 3)
plt.hist(normal, bins = int(np.sqrt(len(normal))),  density = True, alpha = 0.5,color = 'green', label = 'Non-anomalous',zorder = 3)
plt.grid(zorder = 1)
plt.xlabel('Reconstruction error', fontsize = 12)
plt.plot([27, 27,27,27], [0.0,0.05,0.2,0.45], linestyle = '--', color = 'black', label='Threshold = 27.1')
plt.ylim(0,0.45)
plt.legend()
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
tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0,1]).ravel()

cm= confusion_matrix(true, pred, labels=[0,1])
print(cm)
print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
print('FPR: ', fp/(fp+tn))
print('FNR: ', fn/(fn+tp))