import pandas as pd
import os
import numpy as np
from common import *
from helpers.dataset_helpers import box_to_labels
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"Script that reads in the annotated files and creates separate files containing the image files for anomalous and non-anomalous images"

np.random.seed(42)
f = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/annotation_testing_backround'
with pd.HDFStore( f,  mode='r') as store:
        db = store.select('db')
        print(f'Reading {DataBaseFileLocation_local+f}')
main_db = db

main_db = main_db.reset_index()

main_db = main_db.drop(main_db[main_db.orig_boxX.map(len) == 0].index)
main_db['Path'] = main_db.Campaign + '/' + main_db.DUT + '/' + main_db.FileName
files = main_db.Path
print('Number of normal files: ', len(files))
print(files)

labels = main_db[['bound_boxX', 'bound_boxY', 'bound_box_dimX', 'bound_box_dimY', 'orig_boxX', 'orig_boxY']].copy()
labels['crop_lbls'] = pd.NaT
labels['crop_lbls'] = labels.apply(lambda x: box_to_labels(x.orig_boxX, x.orig_boxY), axis=1)

## split into training and test sets
base_dir = '/db/'
dir_ae = "AE/"
dir_det = "DET/"
dirs_ae = os.listdir(base_dir + dir_ae)
dirs_det = os.listdir(base_dir + dir_det)

def save_file(path, nplist, overwrite):
    if overwrite:
        np.save(path, nplist)
        print('Saving...', path)
    else:
        if not os.path.exists(path):
            np.save(path, nplist)
            print('Saving...', path)

save_file(base_dir + dir_ae + 'X_bgrnd.npy', files.values, True)
save_file(base_dir + dir_ae + 'Y_bgrnd.npy', labels.crop_lbls.values, True)



