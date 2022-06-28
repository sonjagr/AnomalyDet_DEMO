import numpy as np
import os
INPUT_DIM = 3
from helpers.dataset_helpers import create_cnn_dataset
from helpers.cnn_helpers import rotate, format_data, bayer2rgb, tf_bayer2rgb, patch_images, flip, plot_metrics, plot_examples, crop, bright_encode_rgb, encode, encode_rgb
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
import random, time, argparse
import tensorflow as tf
import cv2

tf.keras.backend.clear_session()

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

## convert bayer to rgb
def bayer2rgb(bayer):
    shape = np.shape(bayer)
    rgb = cv2.cvtColor(bayer.reshape(shape[0],shape[1],1).astype('uint8'), cv2.COLOR_BAYER_RG2RGB)
    return rgb

ae = AutoEncoder()
print('Loading autoencoder and data...')
ae.load('saved_class/model_AE_TQ3_500_to_500_epochs')
#ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')

base_dir = 'db/'
dir_det = 'DET/'
dir_ae = 'AE/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
images_dir_loc = 'F:/ScratchDetection/MeasurementCampaigns/'

## extract normal images for training
X_train_list_ae = np.load(base_dir + dir_ae + 'X_train_AE.npy', allow_pickle=True)
X_test_list_ae = np.load(base_dir + dir_ae + 'X_test_AE.npy', allow_pickle=True)

X_test_list_ae = [images_dir_loc + s for s in X_test_list_ae]

images = X_test_list_ae[:5]

image = np.load(images[1])
plt.imshow(image, cmap="gray")
plt.show()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
#gray  = cv2.bitwise_not(gray)
edges = cv2.Canny(gray,0,50, apertureSize=3)
plt.imshow(edges, cmap="gray")
plt.show()

'''
img_gray  = cv2.bitwise_not(img_gray)
_, img_thr = cv2.threshold(img_gray, thresh=150, maxval=255, type=cv2.THRESH_BINARY)
plt.imshow(img_thr, cmap="gray")
plt.show()
fig, axs = plt.subplots(2, 1)
axs[0].set_title("Thresholded")
axs[0].imshow(img_thr, cmap="gray")
# Find lines
lines = cv2.HoughLinesP(img_thr, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
lines = lines.squeeze()
axs[1].set_title("Grayscale with Lines")
axs[1].imshow(img_gray, cmap="gray")
for x1, y1, x2, y2 in lines:
    axs[1].plot([x1, x2], [y1, y2], "r")
fig.show()
'''