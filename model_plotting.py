import numpy as np
import tensorflow as tf
import os, sys, math, time
import pickle, argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from helpers.dataset_helpers import create_dataset, create_cnn_dataset, box_index_to_coords
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from helpers.dataset_helpers import create_dataset
from autoencoders2 import *
from common import *
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
ae = AutoEncoder()

def visualize_model(model):
    #model.save('/home/gsonja/Desktop/models/')
    plot_model(model, to_file='/home/gsonja/Desktop/models/model.png')

import cv2
def plot_ae(original, aed, i):
    import matplotlib
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    original_rgb = cv2.cvtColor(original.astype('uint8'), cv2.COLOR_BAYER_RG2RGB)

    ax1.imshow(original_rgb, vmin=0, vmax=255)
    #for i in i_a:
    #    box = box_index_to_coords(i)
    #    rec = matplotlib.patches.Rectangle((int(box[0]), int(box[1])), 160, 160, linewidth=1, edgecolor='r', facecolor='none')
    #    ax[0].add_patch(rec)
    ax1.set_title('Original', fontsize =16)
    ax1.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    aed_rgb = cv2.cvtColor(aed.astype('uint8'), cv2.COLOR_BAYER_RG2RGB)
    ax2.imshow(aed_rgb, vmin=0, vmax=255)
    ax2.set_title('Auto-encoded', fontsize =16)
    ax2.tick_params(axis='both', which='both', bottom=False, left = False, labelbottom=False,labelleft=False)

    cmap = 'rainbow'
    diff = np.abs(np.subtract(aed,original))
    print(np.min(diff), np.max(diff))
    ax3.imshow(diff,cmap = cmap)
    ax3.set_title('Difference', fontsize =16)
    ax3.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    plt.tight_layout()
    #plt.savefig('aed_imgs/aed_example_nopatch_%i.png' % i,dpi=600)
    plt.show()

def plot_ae_zoom(original, aed, i, boxX, boxY, lower, times):

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    extent = (0, 3840, 0, 2720)
    original_rgb = cv2.cvtColor(original.astype('uint8'), cv2.COLOR_BAYER_RG2RGB)
    ax1.imshow(original_rgb, extent=extent, origin="upper", vmin=0, vmax=255)
    ax1.set_title('Original', fontsize =16)

    x1, x2, y1, y2 = boxX[0], boxX[1], boxY[0], boxY[1]
    xdim = (3840/(x2-x1)*times)/100
    ydim = (2720/(y2-y1)*times)/100
    print('xdimm', xdim)
    axins = ax1.inset_axes([lower[0], lower[1], xdim, ydim])
    axins.imshow(original_rgb,extent=(0, 3840, 0, 2720), origin="upper", vmin=0, vmax=255)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.spines['top'].set_color('red')
    axins.spines['bottom'].set_color('red')
    axins.spines['left'].set_color('red')
    axins.spines['right'].set_color('red')
    axins.set_xticks([])
    axins.set_yticks([])
    ax1.indicate_inset_zoom(axins, edgecolor="red")
    ax1.tick_params(axis='both', which='both', bottom=False, left = False, labelbottom=False,labelleft=False)

    aed_rgb = cv2.cvtColor(aed.astype('uint8'), cv2.COLOR_BAYER_RG2RGB)
    ax2.imshow(aed_rgb, extent=extent, origin="upper", vmin=0, vmax=255)
    ax2.set_title('Auto-encoded', fontsize =16)
    ax2.tick_params(axis='both', which='both', bottom=False, left = False, labelbottom=False,labelleft=False)

    axins = ax2.inset_axes([lower[0], lower[1], xdim, ydim])
    axins.imshow(aed_rgb, extent=extent, origin="upper", vmin=0, vmax=255)
    x1, x2, y1, y2 = boxX[0], boxX[1], boxY[0], boxY[1]
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.spines['top'].set_color('red')
    axins.spines['bottom'].set_color('red')
    axins.spines['left'].set_color('red')
    axins.spines['right'].set_color('red')
    axins.set_xticks([])
    axins.set_yticks([])
    ax2.indicate_inset_zoom(axins, edgecolor="red")

    cmap = 'rainbow'
    diff = np.sqrt((aed-original)**2)
    ax3.imshow(diff,cmap = cmap, vmin=50, extent=extent, origin="upper")
    ax3.set_title('Difference', fontsize =16)
    ax3.tick_params(axis='both', which='both', bottom=False, left = False, labelbottom=False,labelleft=False)

    axins = ax3.inset_axes([lower[0], lower[1], xdim, ydim])
    axins.imshow(diff, extent=extent, cmap = cmap, vmin=50,  origin="upper")
    x1, x2, y1, y2 = boxX[0], boxX[1], boxY[0], boxY[1]
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.spines['top'].set_color('red')
    axins.spines['bottom'].set_color('red')
    axins.spines['left'].set_color('red')
    axins.spines['right'].set_color('red')
    axins.set_xticks([])
    axins.set_yticks([])
    ax3.indicate_inset_zoom(axins, edgecolor="red")
    plt.tight_layout()
    i = 2
    #plt.savefig('aed_imgs/aed_example_nopatch_zoom_%i.png' % i,dpi=600)
    plt.show()

def comparison_plot(checkpoint_loc, list_of_models, colors=['r', 'b' ,'g', 'orange']):
    list_of_labels = ['TQ3','TQ3_DO_smaller_lr','TQ3_DO','SG3']
    plt.figure(1, figsize=(8, 5))
    for i in range(0, len(list_of_models)):
        with open(checkpoint_loc +'/'+ str(list_of_models[i] + '/cost.pkl'), 'rb') as f:
            x = pickle.load(f)
        test_losses = x['test_losses']
        n=5
        train_losses = x['train_losses']
        print(len(train_losses))
        min_steps_test = x['min_steps_test']
        #plt.plot(np.arange(1, len(test_losses) + 1, 1/8000), train_losses, c='y', label = 'Train', linestyle='--')
        plt.plot(np.arange(1, len(test_losses) + 1, 1), test_losses, color = 'C1', label='Validation')
    plt.ylabel('L2 loss', fontsize = 14)
    plt.xlabel('Epoch', fontsize = 14)
    #plt.title('Validation loss during training', fontsize = 16)
    plt.grid(zorder=-3)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend( fontsize = 14)
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/' + 'final_ae' + '_val_loss.png', dpi = 600)
    plt.show()

checkpoints_loc = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints'
list_of_models = ['TQ3_1_cont']

base_dir = 'db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET.npy', allow_pickle=True).tolist()
def_dataset = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=False)


def plot_aed_zoom(plotting_dataset, boxX, boxY, lower, times):
    ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1/model_AE_TQ3_5_to_5_epochs')
    p=0
    for x_plot, y_plot in plotting_dataset:
        img = x_plot
        y_plot = y_plot.numpy().reshape(17 * 24)
        img = img[0].numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
        encoded_img = ae.encode(img)
        decoded_img = ae.decode(encoded_img)
        plot_ae_zoom(img.reshape(2720, 3840), decoded_img.numpy().reshape(2720, 3840), p, boxX[p], boxY[p], lower[p], times)
        p = p+1

def plot_aed(plotting_dataset, name):
    ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')
    p=0
    for x_plot, y_plot in plotting_dataset:
        img = x_plot
        y_plot = y_plot.numpy().reshape(17 * 24)
        img = img[0].numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
        encoded_img = ae.encode(img)
        decoded_img = ae.decode(encoded_img)
        plot_ae(img.reshape(2720, 3840), decoded_img.numpy().reshape(2720, 3840), p)
        p = p+1

def classifier_scores(test_loss, train_loss):
    plt.plot(np.arange(0,len(test_loss)), test_loss, label = 'Test loss')
    plt.plot(np.arange(0,len(train_loss)), train_loss, linestyle= '--', label = 'Train loss')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Binary cross-entropy')
    plt.show()

mnbr = 'testing_all_data'
test_loss = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/losses/%s/test_loss_%s.npy' % (mnbr,mnbr))
train_loss = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/losses/%s/train_loss_%s.npy' % (mnbr,mnbr))
#classifier_scores(test_loss, train_loss)
#comparison_plot(checkpoints_loc, list_of_models)

plotting_dataset = def_dataset.shuffle(100, seed = 1).take(2)
#plot_aed(plotting_dataset, 1)

boxX = [(3000,3600),(500,1100),(1000,1500)]
boxY = [(0,300),(0,400),(0,400)]
lower = [(0.55,0.3),(0.3,0.2),(0.55,0.3)]
times = 7
plot_aed_zoom(plotting_dataset, boxX, boxY, lower, times)

boxX = [(3000,3500)]
boxY = [(0,300)]
lower = [(0.55,0.3)]
times = 5
#plotting_dataset = def_dataset.shuffle(100, seed = 2).take(5)
plotting_dataset = plotting_dataset.take(1)
#plot_aed_zoom(plotting_dataset, boxX, boxY, lower, times)
#plotting_dataset = plotting_dataset.skip(4)
boxX = [(1500,1800)]
boxY = [(200,500)]
lower = [(0.2,0.35)]
times = 6

#plot_aed_zoom(plotting_dataset, boxX, boxY, lower, times)

#plot_aed(plotting_dataset, 2)

