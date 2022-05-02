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

MTN = 1
train_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/' +'train_img_list_%i.npy' % MTN)
train_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/' +'train_lbl_list_%i.npy' % MTN)
val_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/' +'val_img_list_%i.npy' % MTN)
val_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/' +'val_lbl_list_%i.npy' % MTN)
test_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/' +'test_img_list_%i.npy' % MTN)
test_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/' +'test_lbl_list_%i.npy' % MTN)

for i in train_img_list:
    minim = np.min(i)
    if minim < 0:
        print(minim)

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

test_pred_plot = [0,0,0,0,0,0]
test_true_plot = train_lbl_list[0:6]
test_img_plot = train_img_list[0:6, :, :]
plot_examples(test_pred_plot, test_true_plot, test_img_plot, '0')
test_pred_plot = [0,0,0,0,0,0]
test_true_plot = train_lbl_list[6:12]
test_img_plot = train_img_list[6:12, :, :]
plot_examples(test_pred_plot, test_true_plot, test_img_plot, '0')
test_true_plot = train_lbl_list[12:18]
test_img_plot = train_img_list[12:18, :, :]
plot_examples(test_pred_plot, test_true_plot, test_img_plot, '0')
test_true_plot = train_lbl_list[18:24]
test_img_plot = train_img_list[18:24, :, :]
plot_examples(test_pred_plot, test_true_plot, test_img_plot, '0')
test_true_plot = train_lbl_list[24:30]
test_img_plot = train_img_list[24:30, :, :]
plot_examples(test_pred_plot, test_true_plot, test_img_plot, '0')
test_true_plot = train_lbl_list[30:36]
test_img_plot = train_img_list[30:36, :, :]
plot_examples(test_pred_plot, test_true_plot, test_img_plot, '0')



u= np.unique(train_img_list, axis  =0)
print(len(train_img_list))
print(len(u))
u= np.unique(test_img_list, axis  =0)
print(len(test_img_list))
print(len(u))
u= np.unique(val_img_list, axis  =0)
print(len(val_img_list))
print(len(u))



maths = False
for i in val_img_list:
    for j in test_img_list:
        if np.array_equal(i, j):
            print('MATHS')
            maths = True
            break

print(maths)