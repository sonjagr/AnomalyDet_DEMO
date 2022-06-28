import os

substring = 'rescan'
count = 0
directory = 'F:/ScratchDetection/MeasurementCampaigns/Fall2021_PM8/'
for dut in os.listdir(directory):
    if 'after' not in dut and 'repeat' not in dut and 'onDF' not in dut:
        for filename in os.listdir(directory + dut):
            if substring in filename:
                count = count + 1
                print(filename)

print(count)

import pandas as pd
f = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/main/Fall2021_PM8.h5'
with pd.HDFStore( f,  mode='r') as store:
        db = store.select('db')
main_db = db

anomalous_db = main_db[main_db.Normal == False]
print(anomalous_db)
anomalous_db = anomalous_db.reindex()

print(anomalous_db.loc[('Fall2021_PM8', '8inch_198ch_N4791_19')])


import numpy as np
base_dir = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db'
dir_det = 'DET/'
dir_ae = 'AE/'
images_dir_loc = 'F:/ScratchDetection/MeasurementCampaigns/'

## extract normal images for training
X_train_list_ae = np.load(os.path.join(base_dir, dir_ae, 'X_train_AE.npy'), allow_pickle=True)
X_test_list_ae = np.load(os.path.join(base_dir, dir_ae, 'X_test_AE.npy'), allow_pickle=True)

np.random.seed(1)
X_train_list_normal_to_remove = np.random.choice(X_train_list_ae, 16000, replace=False)
X_test_val_list_normal_to_remove = np.random.choice(X_test_list_ae, 4000, replace=False)

X_train_list_normal_removed = [x for x in X_train_list_ae if x not in X_train_list_normal_to_remove]
X_test_list_normal_removed = [x for x in X_test_list_ae if x not in X_test_val_list_normal_to_remove]

X_train_normal_list = np.random.choice(X_train_list_normal_removed, 1000, replace=False)
X_val_normal_list = np.random.choice(X_test_list_normal_removed, 1000, replace=False)
X_train_normal_list = [images_dir_loc + s for s in X_train_normal_list]
X_val_normal_list = [images_dir_loc + s for s in X_val_normal_list]

import matplotlib.pyplot as plt
for i in X_train_normal_list:
    plt.imshow(np.load(i))
    plt.show()

