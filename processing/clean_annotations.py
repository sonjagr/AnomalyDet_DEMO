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

@tf.function
def process_crop(image, label):
    image, label = crop(image, label)
    return image, label

#Y_train_det_list_2 = np.load(os.path.join(base_dir, dir_det , 'Y_train_DET_final_cleaned.npy'), allow_pickle=True).tolist()
#print(len(Y_train_det_list_2)/408)


## load the images containing anomalies
#X_train_det_list = np.load(os.path.join(base_dir, dir_det, 'X_train_DET_final.npy'), allow_pickle=True)
X_val_det_list = np.load(os.path.join(base_dir, dir_det, 'X_test_DET_final.npy'), allow_pickle=True)

#X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_val_det_list = [images_dir_loc + s for s in X_val_det_list]

#Y_train_det_list = np.load(os.path.join(base_dir, dir_det , 'Y_train_DET_final_cleaned.npy'), allow_pickle=True).tolist()
Y_val_det_list = np.load(os.path.join(base_dir , dir_det , 'Y_test_DET_final.npy'), allow_pickle=True).tolist()


#train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=False)
val_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list, _shuffle=False)

#train_ds = train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#train_ds = train_ds.flat_map(patch_images).unbatch()
val_ds = val_ds.flat_map(patch_images).unbatch()

import matplotlib.pyplot as plt
#Y_train_det_list = np.array(Y_train_det_list).flatten()
#Y_train_det_list_new = []
Y_val_det_list = np.array(Y_val_det_list).flatten()
Y_val_det_list_new = []
i= 0
from tqdm import tqdm
for x, y in tqdm(val_ds, total=200*408):
    y = y.numpy()
    if Y_val_det_list[i] == 1.:
        x = x.numpy().reshape(160,160)
        plt.imshow(x)
        plt.show()
        inp = input('p to remove, enter to keep')
        if inp == 'p':
            Y_val_det_list_new = np.append(Y_val_det_list_new, 0)
        elif inp != 'p':
            Y_val_det_list_new = np.append(Y_val_det_list_new, 1)
    else:
        Y_val_det_list_new = np.append(Y_val_det_list_new, 0)

    i = i+1

    Y_val_det_list_new = np.array(Y_val_det_list_new)
    #save_file(base_dir + dir_det + 'Y_test_DET_final_cleaned.npy', Y_val_det_list_new,  True)
