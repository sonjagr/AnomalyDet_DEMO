import os
import pickle
pickle.HIGHEST_PROTOCOL = 4
import pandas as pd
from helpers.dataset_helpers import create_cnn_dataset, process_anomalous_df_to_numpy
from common import *

base_dir = TrainDir_pc
images_dir_loc = imgDir_pc
dir_det = '../db/DET/'
dir_ae = '../db/AE/'
f = os.path.join(base_dir, '../db/TRAIN_DATABASE_20220711')
with pd.HDFStore( f,  mode='r') as store:
        final= store.select('db')
        print(f'Reading {DataBaseFileLocation_local+f}')

X_train, Y_train= process_anomalous_df_to_numpy(final)
print(len(X_train), X_train[0])

f = os.path.join(base_dir, '../db/TEST_DATABASE_20220711')
with pd.HDFStore( f,  mode='r') as store:
        test_db = store.select('db')
        print(f'Reading {DataBaseFileLocation_local+f}')
test_db['Date'] = pd.to_datetime('today').date()
X_test, Y_test = process_anomalous_df_to_numpy(test_db)
print(len(X_test), X_test[0])
print(any(i in X_train for i in X_test))

import numpy as np

X_train_norm = np.append(np.load(os.path.join(base_dir, dir_ae, 'X_train_forcnn_clean.npy'), allow_pickle=True), np.load(os.path.join(base_dir, dir_ae, 'X_train_forcnn_clean_2.npy'), allow_pickle=True))
X_val_norm = np.append(np.load(os.path.join(base_dir, dir_ae, 'X_test_forcnn_clean.npy'), allow_pickle=True), np.load(os.path.join(base_dir, dir_ae, 'X_test_forcnn_clean_2.npy'), allow_pickle=True))
X_train_norm  = [s.replace('F:/ScratchDetection/MeasurementCampaigns/', '') for s in X_train_norm ]
X_val_norm  = [s.replace('F:/ScratchDetection/MeasurementCampaigns/', '') for s in X_val_norm ]

print(X_train_norm[0])
print(X_val_norm[0])

print(bool(set(X_train_norm) & set(X_val_norm)))

#np.save('NORMAL_TRAIN_20220711.npy', X_train_norm)
#np.save('NORMAL_TEST_20220711.npy', X_val_norm)




