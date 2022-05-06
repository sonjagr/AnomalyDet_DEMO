import numpy as np
import tensorflow
import os, sys, math, time
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from tensorflow import keras
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from helpers.dataset_helpers import create_dataset
from autoencoders import *
from common import *

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_ae = "AE/"
dir_det = "DET/"

X_test_list = np.load(base_dir + dir_ae + 'X_test_AE.npy', allow_pickle=True)
X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)
np.random.seed(42)
X_test_val_list = np.random.choice(X_test_list, 2000, replace=False)
X_test_list, X_val_list = train_test_split(X_test_val_list, test_size=0.5, random_state=42)

X_val_list_norm = [imgDir_gpu + s for s in X_val_list][5:8]
X_val_list_det = [imgDir_gpu + s for s in X_test_det_list][5:8]
X_val_list = X_val_list_norm + X_val_list_det

batch_size = 1

def resize(item):
    model = tf.keras.Sequential([tf.keras.layers.Cropping2D(cropping=((0, 16),(0, 0)))])
    return model(item)

val_dataset = create_dataset(X_val_list, _shuffle=True).batch(batch_size)

xshape = 2736
#val_dataset = val_dataset.map(lambda item: tuple(tf.py_function(resize, [item], [tf.float32,])))

model = 'TQ3'
savemodel = 'TQ3_1_cont'
epoch = '500'
print('model: ', model , 'savemodel: ', savemodel, 'epoch: ', epoch)

saveloc = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/'+savemodel
Path(saveloc).mkdir(parents=True, exist_ok=True)

ae = AutoEncoder()
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/'+savemodel+'/model_AE_'+model+'_'+epoch+'_to_'+epoch+'_epochs')
#ae.build(input_shape=(PICTURESIZE_Y, PICTURESIZE_X, 1))

print(ae.encoder.summary())
print(ae.decoder.summary())

with open('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/'+savemodel+'/cost.pkl', 'rb') as f:
    x = pickle.load(f)
test_losses = x['test_losses']
min_steps_test = x['min_steps_test']
plt.plot(np.arange(1,len(test_losses)+1,1), test_losses)
plt.ylabel('L loss (L2)')
plt.xlabel('Epoch')
plt.grid(zorder = -3)
plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/'+savemodel+'/test_loss.png')
plt.show()

import cv2
import matplotlib.pyplot as plt
def plot_image(image, i):
    image = image[0]
    fig, ax = plt.subplots()
    ax.imshow(image, vmin=0, vmax=255)
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/'+savemodel+'/aed_img_%s.png' % i)
    plt.show()
    return image

def plot_orig_image(image, i):
    image = np.load(image)
    fig, ax = plt.subplots()
    ax.imshow(image, vmin=0, vmax=255)
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/'+savemodel+'/orig_img_%s.png' % i)
    plt.show()
    return image[:xshape, :]

def plot_diff(image, i):
    fig, ax = plt.subplots()
    ax.imshow(image, vmin=0, vmax=255)
    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/'+savemodel+'/diff_2_img_%s.png' % i)
    plt.show()

i=0
for x in tqdm(val_dataset, total=10):
    val_img =X_val_list[i]
    orig_img = plot_orig_image(val_img, i)
    val_enc = ae.encode(x)
    val_dec = ae.decode(val_enc)
    dec_img = plot_image(val_dec, i)
    diff = np.sqrt((tf.reshape(dec_img, [xshape,3840]) - orig_img)**2)
    diff = (tf.reshape(dec_img, [xshape, 3840]) - orig_img)
    plot_diff(diff, i)
    i = i + 1