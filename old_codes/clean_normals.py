import pandas as pd
import os
import numpy as np
from common import *
import numpy as np
from helpers.dataset_helpers import create_cnn_dataset
from helpers.cnn_helpers import rotate, format_data, patch_images, flip, plot_metrics, plot_examples, crop, bright_encode, encode, encode_rgb, bright_encode_rgb, tf_bayer2rgb
from old_codes.autoencoders import *
import random, time, argparse, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

"Script that reads in the annotated files and creates separate files containing the image files for anomalous and non-anomalous images"

base_dir = TrainDir_pc
images_dir_loc = imgDir_pc
dir_det = '../db/DET/'
dir_ae = '../db/AE/'

def save_file(path, nplist, overwrite):
    if overwrite:
        np.save(path, nplist)
        #print('Saving...', path)
    else:
        if not os.path.exists(path):
            np.save(path, nplist)
            #print('Saving...', path)

random.seed(42)
base_dir = TrainDir_pc
images_dir_loc = imgDir_pc
dir_det = '../db/DET/'
dir_ae = '../db/AE/'

## extract normal images for training
X_train_list_ae = np.load(os.path.join(base_dir, dir_ae, ''), allow_pickle=True)
X_test_list_ae = np.load(os.path.join(base_dir, dir_ae, ''), allow_pickle=True)
print(len(X_train_list_ae), len(X_test_list_ae))


N_det_train = 2000
N_det_val = 420

## you can choose up to 4496/1125 whole images

np.random.seed(42)
X_train_normal_list = np.random.choice(X_train_list_ae, int(N_det_train),  replace=False)
X_val_normal_list = np.random.choice(X_test_list_ae, int(N_det_val),  replace=False)

X_train_removed = [x for x in X_train_list_ae if x not in X_train_normal_list]
X_val_removed = [x for x in X_test_list_ae if x not in X_val_normal_list]

X_train_normal_list = [images_dir_loc + s for s in X_train_removed]
X_val_normal_list = [images_dir_loc + s for s in X_val_removed ]

old_dirty = np.load(base_dir + dir_ae + '')
old_clean = np.load(base_dir + dir_ae + '')
print(len(old_dirty))
#old_dirty = [images_dir_loc + s for s in old_dirty]
#old_clean = [images_dir_loc + s for s in old_clean]

old = np.append(old_clean, old_dirty)
print(len(old))
X_train_normal_list = [x for x in X_train_normal_list if x not in old]

import matplotlib.pyplot as plt

X_list_new_clean = old_clean
X_list_new_dirty = old_dirty
i= 0
from tqdm import tqdm
for x in tqdm(X_train_normal_list, total=len(X_train_normal_list)):
    img = np.load(x)
    img = img.reshape(2736,3840)
    plt.imshow(img)
    plt.show()
    inp = input('p to remove, enter to keep')
    if inp == 'p':
        print('removed')
        X_list_new_dirty = np.append(X_list_new_dirty, x)
    elif inp != 'p':
        X_list_new_clean = np.append(X_list_new_clean, x)
    i = i+1

    X_list_new_dirty = np.array(X_list_new_dirty)
    X_list_new_clean = np.array(X_list_new_clean)
    #save_file(base_dir + dir_ae + 'X_train_forcnn_clean_2.npy', X_list_new_clean,  True)
    #save_file(base_dir + dir_ae + 'X_train_forcnn_toann_2.npy', X_list_new_dirty, True)
