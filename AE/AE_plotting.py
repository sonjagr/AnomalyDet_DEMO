import numpy as np
import pickle
from helpers.dataset_helpers import  create_cnn_dataset
from autoencoders2 import *
from common import *
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
ae = AutoEncoder()
import cv2

def losses_plot(checkpoint_loc, model_name, save= False, saveloc = 'None'):
    plt.figure(1, figsize=(8, 5))
    with open(checkpoint_loc, 'rb') as f:
        x = pickle.load(f)
    test_losses = x['test_losses']
    train_losses = x['train_losses']
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
    original_rgb = cv2.cvtColor(original.astype('uint8'), cv2.COLOR_BAYER_RG2RGB)
    cmap = 'gray_r'
    ax1.imshow(original, cmap = cmap)
    #for i in i_a:
    #    box = box_index_to_coords(i)
    #    rec = matplotlib.patches.Rectangle((int(box[0]), int(box[1])), 160, 160, linewidth=1, edgecolor='r', facecolor='none')
    #    ax[0].add_patch(rec)
    #ax1.set_title('Original', fontsize =16)
    ax1.tick_params(axis='both', which='both', bottom=False,left = False,labelbottom=False,labelleft=False)

    aed_rgb = cv2.cvtColor(aed.astype('uint8'), cv2.COLOR_BAYER_RG2RGB)
    ax2.imshow(aed, cmap = cmap, vmin=0, vmax=255)
    #ax2.set_title('Auto-encoded', fontsize =16)
    ax2.tick_params(axis='both', which='both', bottom=False, left = False, labelbottom=False,labelleft=False)

    cmap = 'tab20c'
    cmap = 'gray_r'
    diff = np.abs(np.subtract(aed,original))
    ax3.imshow(diff,  cmap = cmap, vmin = -1, vmax = 180,)
    #ax3.set_title('Difference', fontsize =16)
    ax3.tick_params(axis='both', which='both', bottom=False,left = False, labelbottom=False, labelleft=False)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(saveloc, 'aed_example_nozoom_%i.png' % i), dpi=DPI)
    plt.show()

def plot_ae_zoom(original, aed, boxX, boxY, lower, times, save = False, i = 1, saveloc = None):

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    extent = (0, 3840, 0, 2720)
    original_rgb = cv2.cvtColor(original.astype('uint8'), cv2.COLOR_BAYER_RG2RGB)
    ax1.imshow(original, extent=extent, origin="upper",  cmap = 'gray_r',  vmin=0, vmax=255)
    #ax1.set_title('Original', fontsize =16)

    x1, x2, y1, y2 = boxX[0], boxX[1], boxY[0], boxY[1]
    xdim = (3840/(x2-x1)*times)/100
    ydim = (2720/(y2-y1)*times)/100
    axins = ax1.inset_axes([lower[0], lower[1], xdim, ydim])
    axins.imshow(original,extent=(0, 3840, 0, 2720), origin="upper", cmap = 'gray_r',  vmin=0, vmax=255)
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
    ax2.imshow(aed, extent=extent, origin="upper", cmap = 'gray_r',  vmin=0, vmax=255)
    #ax2.set_title('Auto-encoded', fontsize =16)
    ax2.tick_params(axis='both', which='both', bottom=False, left = False, labelbottom=False,labelleft=False)

    axins = ax2.inset_axes([lower[0], lower[1], xdim, ydim])
    axins.imshow(aed, extent=extent, origin="upper", cmap = 'gray_r', vmin=0, vmax=255)
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

    cmap = 'tab20c'
    cmap = 'gray_r'
    diff = np.sqrt((aed-original)**2)
    ax3.imshow(diff,cmap = cmap, vmin = -1,  vmax =180,extent=extent, origin="upper")
    #ax3.set_title('Difference', fontsize =16)
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
        y_plot = y_plot.numpy().reshape(17 * 24)
        img = img[0].numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
        img = np.multiply(img, 0.75)
        encoded_img = ae.encode(img)
        decoded_img = ae.decode(encoded_img)
        plot_ae_zoom(img.reshape(2720, 3840), decoded_img.numpy().reshape(2720, 3840), boxX[p], boxY[p], lower[p], times, save = save, i = p, saveloc = saveloc)
        p = p+1

def plot_aed(path_to_ae, plotting_dataset, save, saveloc):
    ae.load(path_to_ae)
    p=0
    for x_plot, y_plot in plotting_dataset:
        print(x_plot)
        img = x_plot
        y_plot = y_plot.numpy().reshape(17 * 24)
        img = img[0].numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
        encoded_img = ae.encode(img)
        decoded_img = ae.decode(encoded_img)
        rec_err = np.mean(np.sqrt((decoded_img-img)**2))
        print('Mean recon error: ', rec_err)
        plot_ae(img.reshape(2720, 3840), decoded_img.numpy().reshape(2720, 3840), save = save, i = p, saveloc = saveloc)
        p = p+1

model_name = 'TQ3_2'
savename = 'TQ3_2_more_params_2'
home_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/'
saveloc = os.path.join(home_dir, 'AE_plots')

path_to_loss_file = os.path.join(home_dir, 'checkpoints/%s_1_%s/cost.pkl' % (model_name,savename))
losses_plot(path_to_loss_file, model_name, save= True, saveloc = saveloc)

epoch = 227
base_dir = '../db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
path_to_ae = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_2_1_TQ3_2_more_params_2/AE_TQ3_2_277_to_277_epochs'
#images_dir_loc = 'F:/ScratchDetection/MeasurementCampaigns/'

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET.npy', allow_pickle=True).tolist()
def_dataset = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=False)

plotting_dataset = def_dataset.shuffle(100, seed = 1).take(5)
plot_aed(path_to_ae, plotting_dataset, True, saveloc)

boxX = [(3000,3600),(500,1100),(1000,1500)]
boxY = [(0,300),(0,400),(0,400)]
lower = [(0.55,0.3),(0.3,0.2),(0.55,0.3)]
times = 6
plot_aed_zoom(path_to_ae, plotting_dataset, boxX, boxY, lower, times, False, None)

boxX = [(3000,3500)]
boxY = [(0,300)]
lower = [(0.55,0.3)]
times = 5
plotting_dataset = def_dataset.shuffle(100, seed = 1).take(5)
plotting_dataset = plotting_dataset.take(12)
#plot_aed_zoom(plotting_dataset, boxX, boxY, lower, times)
plotting_dataset = plotting_dataset.skip(4)
boxX = [(1500,1800)]
boxY = [(200,500)]
lower = [(0.2,0.35)]
times = 6

#plot_aed_zoom(plotting_dataset, boxX, boxY, lower, times)



