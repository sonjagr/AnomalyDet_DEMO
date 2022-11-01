## for plotting images before and after autoencoding
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import pickle
import matplotlib.pyplot as plt

from helpers.dataset_helpers import create_cnn_dataset
from autoencoders2 import *
from common import *

ae = AutoEncoder()

## plot training and validation loss
def losses_plot(checkpoint_loc, model_name, save= False, saveloc = 'None'):
    plt.figure(1, figsize=(8, 5))
    with open(checkpoint_loc, 'rb') as f:
        x = pickle.load(f)
    test_losses = x['test_losses']
    #train_losses = x['train_losses']
    #plt.plot(np.arange(1, len(test_losses) + 1, 1/(len(train_losses)/len(test_losses))), train_losses, c='y', label = 'Training', linestyle='--')
    plt.plot(np.arange(1, len(test_losses) + 1, 1), test_losses, color = 'C1', label='Validation')
    plt.ylabel('L2 loss', fontsize = 14)
    plt.xlabel('Epoch', fontsize = 14)
    #plt.title('Validation loss during training', fontsize = 16)
    plt.grid(zorder=-3)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend( fontsize = 14)
    if save:
        plt.savefig(os.path.join(saveloc, 'AE_%s_val_loss.png' % model_name), dpi =DPI)
    plt.show()

def plot_ae(original, aed, save = False, i = 1, saveloc = None):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    cmap = 'gray_r'
    ax1.imshow(original, cmap = cmap)
    ax1.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)
    ax2.imshow(aed, cmap = cmap, vmin=0, vmax=255)
    ax2.tick_params(axis='both', which='both', bottom=False, left = False, labelbottom=False,labelleft=False)

    cmap = 'gray_r'
    diff = np.abs(np.subtract(aed,original))
    ax3.imshow(diff,  cmap = cmap, vmin = -1, vmax = 180,)
    ax3.tick_params(axis='both', which='both', bottom=False,left = False, labelbottom=False, labelleft=False)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(saveloc, 'aed_example_nozoom_%i.png' % i), dpi=DPI)
    plt.show()

def plot_ae_zoom(original, aed, boxX, boxY, lower, times, save = False, i = 1, saveloc = None):

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    extent = (0, PICTURESIZE_X, 0, PICTURESIZE_Y)
    ax1.imshow(original, extent=extent, origin="upper",  cmap = 'gray_r',  vmin=0, vmax=EIGHTBITMAX)

    x1, x2, y1, y2 = boxX[0], boxX[1], boxY[0], boxY[1]
    xdim = (PICTURESIZE_X/(x2-x1)*times)/100
    ydim = (PICTURESIZE_Y/(y2-y1)*times)/100
    axins = ax1.inset_axes([lower[0], lower[1], xdim, ydim])
    axins.imshow(original,extent=(0, PICTURESIZE_X, 0, PICTURESIZE_Y), origin="upper", cmap = 'gray_r',  vmin=0, vmax=EIGHTBITMAX)
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

    ax2.imshow(aed, extent=extent, origin="upper", cmap = 'gray_r',  vmin=0, vmax=EIGHTBITMAX)
    ax2.tick_params(axis='both', which='both', bottom=False, left = False, labelbottom=False,labelleft=False)

    axins = ax2.inset_axes([lower[0], lower[1], xdim, ydim])
    axins.imshow(aed, extent=extent, origin="upper", cmap = 'gray_r', vmin=0, vmax=EIGHTBITMAX)
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

    cmap = 'gray_r'
    diff = np.sqrt((aed-original)**2)
    ax3.imshow(diff,cmap = cmap, vmin = -1,  vmax =180, extent=extent, origin="upper")
    ax3.tick_params(axis='both', which='both', bottom=False, left = False, labelbottom=False, labelleft=False)

    axins = ax3.inset_axes([lower[0], lower[1], xdim, ydim])
    axins.imshow(diff, extent=extent, cmap = cmap, vmin = -1, vmax = 180, origin="upper")
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

    if save:
        plt.savefig(os.path.join(saveloc, 'aed_example_zoom_%i.png' % i), dpi=DPI)
    plt.show()

def plot_aed_zoom(path_to_ae, plotting_dataset, boxX, boxY, lower, times, save, saveloc):
    ae.load(path_to_ae)
    p=0
    for x_plot, y_plot in plotting_dataset:
        img = x_plot
        img = img[0].numpy().reshape(1, 2736, PICTURESIZE_X, 1)[:, :PICTURESIZE_Y, :, :]
        img = np.multiply(img, 0.75)
        encoded_img = ae.encode(img)
        decoded_img = ae.decode(encoded_img)
        plot_ae_zoom(img.reshape(PICTURESIZE_Y, PICTURESIZE_X), decoded_img.numpy().reshape(PICTURESIZE_Y, PICTURESIZE_X), boxX[p], boxY[p], lower[p], times, save = save, i = p, saveloc = saveloc)
        p = p+1

def plot_aed(path_to_ae, plotting_dataset, save, saveloc):
    ae.load(path_to_ae)
    p=0
    for x_plot, y_plot in plotting_dataset:
        print(x_plot)
        img = x_plot
        img = img[0].numpy().reshape(1, 2736, PICTURESIZE_X, 1)[:, :PICTURESIZE_Y, :, :]
        encoded_img = ae.encode(img)
        decoded_img = ae.decode(encoded_img)
        plot_ae(img.reshape(PICTURESIZE_Y, PICTURESIZE_X), decoded_img.numpy().reshape(PICTURESIZE_Y, PICTURESIZE_X), save = save, i = p, saveloc = saveloc)
        p = p+1

saveloc = ''
base_dir = '../db/'
dir_det = 'DET/'
#images_dir_loc = 'F:/ScratchDetection/MeasurementCampaigns/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
path_to_ae = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_2_1_TQ3_2_more_params_2/AE_TQ3_2_277_to_277_epochs'

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET.npy', allow_pickle=True).tolist()
def_dataset = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=False)

plotting_dataset = def_dataset.shuffle(100, seed = 1).take(5)
plot_aed(path_to_ae, plotting_dataset, True, saveloc)