
import pickle
pickle.HIGHEST_PROTOCOL = 4
import pandas as pd
import os
import numpy as np
from helpers.dataset_helpers import create_cnn_dataset, process_anomalous_df_to_numpy
from common import *
base_dir = DataBaseFileLocation_local
dir_det = 'DET/'
dir_ae = 'AE/'
images_dir_loc = imgDir_pc

## extract normal images for training
X_train = np.load(os.path.join(base_dir, dir_det, 'X_train_DET_backround.npy'), allow_pickle=True)
Y_train = np.load(os.path.join(base_dir, dir_det, 'Y_train_DET_backround.npy'), allow_pickle=True).tolist()
print(X_train)

X_test = np.load(os.path.join(base_dir, dir_det, 'X_test_DET_backround.npy'), allow_pickle=True)
Y_test = np.load(os.path.join(base_dir, dir_det, 'Y_test_DET_backround.npy'), allow_pickle=True).tolist()
X_test = [images_dir_loc + s for s in X_test]

X_br = np.append(X_train, X_test)
Y_br = Y_train + Y_test
print(len(X_br), len(Y_br))

db = []
ind = 0
camp, dut, fn, steps = [], [], [], []
for i in X_br:
    _campaign, _dut, _fn = i.split('/')[-3:]
    _s = i.split('step')[-1].replace('.npy', '')
    db.append((_campaign, _dut, _s, _fn, Y_br[ind]))
    ind = ind +1

br_db = pd.DataFrame(db, columns=['Campaign','DUT','Step','FileName','Y'])
br_db['Path'] = br_db.Campaign + '/' + br_db.DUT + '/' + br_db.FileName

br_files = br_db['Path'].to_numpy()
print(br_db)

##make sure these duts are not in training data for br detector
X_list_norm = np.load(os.path.join(base_dir, 'NORMAL_TEST_20220711.npy'), allow_pickle=True)
#X_test_norm_list = [images_dir_loc + s for s in X_test_norm_list]

f = os.path.join(base_dir, 'TEST_DATABASE_20220711')
with pd.HDFStore( f,  mode='r') as store:
        test_db = store.select('db')
        print(f'Reading {DataBaseFileLocation_local+f}')
X_test_det_list, Y_test_det_list = process_anomalous_df_to_numpy(test_db)

X_list_test_for_CNN = np.append(X_list_norm, X_test_det_list)

br_files_removed = np.setdiff1d(br_files, X_list_test_for_CNN)
print(len(br_files_removed))

br_db_train = br_db[~br_db['Path'].isin(X_list_test_for_CNN)]
br_db_test = br_db.drop(br_db_train.index)
print(br_db_train)
print(br_db_test)

X_br_train = br_db_train['Path'].to_numpy()
Y_br_train = br_db_train['Y'].to_numpy()

X_br_test = br_db_test['Path'].to_numpy()
Y_br_test = br_db_test['Y'].to_numpy()

np.save(os.path.join(base_dir, dir_det, 'X_train_DET_backround_20220711.npy'), X_br_train)
np.save(os.path.join(base_dir, dir_det, 'Y_train_DET_backround_20220711.npy'), Y_br_train)

np.save(os.path.join(base_dir, dir_det, 'X_test_DET_backround_20220711.npy'), X_br_test)
np.save(os.path.join(base_dir, dir_det, 'Y_test_DET_backround_20220711.npy'), Y_br_test)

