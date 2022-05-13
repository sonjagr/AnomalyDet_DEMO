import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
import cv2
from helpers.dataset_helpers import create_cnn_dataset, create_dataset
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import random

random.seed(42)

from helpers.dataset_helpers import rotate_img, rgb2bayer, bayer2rgb, change_brightness_patch, change_brightness
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

ae = AutoEncoder()
print('Loading autoencoder...')
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)

X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_test_det_list = [images_dir_loc + s for s in X_test_det_list]

Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET.npy', allow_pickle=True).tolist()
Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()

X_train_det_list = X_train_det_list[:2]
X_test_det_list = X_test_det_list[:2]

N_det_test = int(len(X_test_det_list)/2)
N_det_train = len(X_train_det_list)
print('N train, N test : ', N_det_train, N_det_test)

def encode(img):
    img = img.numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img).numpy()
    aed_img = np.sqrt(np.power(np.subtract(img, decoded_img),2))
    return aed_img

def crop(img):
    img = img.numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
    return img

## split image into patches
def split(img):
    img = img.numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
    img = tf.convert_to_tensor(img, np.float32)
    split_img = tf.image.extract_patches(images=img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1], rates=[1, 1, 1, 1], padding='VALID')
    return split_img.numpy().reshape(17 * 24, 160 * 160)

imgs = create_dataset(X_train_det_list)
cropped_imgs = imgs.map(lambda x: tuple(tf.py_function(crop, [x], [tf.float32, ])))
diff_imgs = imgs.map(lambda x: tuple(tf.py_function(encode, [x], [tf.float32, ])))

def crop_images(image):
    split_img = tf.image.extract_patches(images=image, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1], rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [17*24,  160 * 160])
    noisy_ds = tf.data.Dataset.from_tensors((re))
    return noisy_ds

print(tf.data.experimental.cardinality(diff_imgs).numpy().shape)
patched_imgs = diff_imgs.flat_map(crop_images)
print(tf.data.experimental.cardinality(patched_imgs).numpy().shape)

#patched_imgs = diff_imgs.flat_map(lambda x: tuple(tf.py_function(crop_images, [x], [tf.float32, ])))

w=0
for X in tqdm(cropped_imgs, total=tf.data.experimental.cardinality(cropped_imgs).numpy()):
    if w == 0:
        plt.imshow(X[0].numpy().reshape(2720, 3840))
        plt.show()
    w = w+1

w=0
for X in tqdm(diff_imgs, total=tf.data.experimental.cardinality(diff_imgs).numpy()):
    if w == 0:
        plt.imshow(X[0].numpy().reshape(2720, 3840))
        plt.show()
    w = w+1

w=0
for X in tqdm(patched_imgs, total=tf.data.experimental.cardinality(patched_imgs).numpy()):
    if w == 0:
        print(X)
        plt.imshow(X[0].numpy().reshape(160, 160))
        print(X[0].numpy().shape)
        plt.show()
        plt.imshow(X[0].numpy().reshape(160, 160))
        plt.show()
    w = w+1

print(tf.data.experimental.cardinality(imgs).numpy())
