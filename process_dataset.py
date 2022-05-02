import pandas as pd
import os
import numpy as np
from common import *
from helpers.dataset_helpers import box_to_labels
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

np.random.seed(42)
f = 'db/three_annotations/main_db_bb_crop.h5'
with pd.HDFStore(DataBaseFileLocation_gpu + f,  mode='r') as store:
        db = store.select('db_bb_crop')
        print(f'Reading {DataBaseFileLocation_local+f}')
main_db = db

## select only normal, ie not anomalous images
normal_db = main_db[main_db.Normal == True]
anomalous_db = main_db[main_db.Normal == False]

normal_db = normal_db.reset_index()
anomalous_db = anomalous_db.reset_index()

## for now, we are dropping one DUT because the path is unusual
normal_db = normal_db.drop(normal_db.loc[(normal_db.Campaign == 'September2021_PM8') & (normal_db.DUT == '8inch_198ch_N3311_7')].index)
normal_db['Path'] = normal_db.Campaign + '/' + normal_db.DUT + '/' + normal_db.FileName
normal_files = normal_db.Path
print('Number of normal files: ', len(normal_files))

anomalous_db = anomalous_db.drop(anomalous_db.loc[(anomalous_db.Campaign == 'September2021_PM8') & (anomalous_db.DUT == '8inch_198ch_N3311_7')].index)
anomalous_db['Path'] = anomalous_db.Campaign + '/' + anomalous_db.DUT + '/' + anomalous_db.FileName
anomalous_files = anomalous_db.Path
print('Number of anomalous files: ', len(anomalous_files))

anomalous_labels = anomalous_db[['bound_boxX', 'bound_boxY', 'bound_box_dimX', 'bound_box_dimY', 'orig_boxX', 'orig_boxY']].copy()
anomalous_labels['crop_lbls'] = pd.NaT
anomalous_labels['crop_lbls'] = anomalous_labels.apply(lambda x: box_to_labels(x.orig_boxX, x.orig_boxY), axis=1)

## split into training and test sets
X_train_normal, X_test_normal = train_test_split(normal_files.values, test_size=0.2, shuffle=True, random_state=42)
X_train_anomalous, X_test_anomalous, Y_train_anomalous, Y_test_anomalous = train_test_split(anomalous_files.values, anomalous_labels.crop_lbls.values, test_size=0.2, shuffle=True, random_state=1)

X_train_normal_list = X_train_normal
X_test_normal_list = X_test_normal

X_train_anomalous_list = X_train_anomalous
X_test_anomalous_list = X_test_anomalous

Y_train_anomalous_list = Y_train_anomalous
Y_test_anomalous_list = Y_test_anomalous

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
#base_dir = 'db/'
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

save_file(base_dir + dir_ae + 'X_train_AE.npy', X_train_normal_list, False)
save_file(base_dir + dir_ae + 'X_test_AE.npy', X_test_normal_list, False)

save_file(base_dir + dir_det + 'X_train_DET.npy', X_train_anomalous_list,  False)
save_file(base_dir + dir_det + 'X_test_DET.npy', X_test_anomalous_list,  False)

save_file(base_dir + dir_det + 'Y_train_DET.npy', Y_train_anomalous_list,  False)
save_file(base_dir + dir_det + 'Y_test_DET.npy', Y_test_anomalous_list, False)

print(f'Number of training samples for AE is {len(X_train_normal_list)} and testing samples {len(X_test_normal_list)}')
print(f'Number of training samples for CNN is {len(X_train_anomalous_list)} and testing samples {len(X_test_anomalous_list)}')


