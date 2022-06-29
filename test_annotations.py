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

#print(count)

import pandas as pd
f = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/three_annotations/main_db_bb_crop.h5'
with pd.HDFStore( f,  mode='r') as store:
    print(store.keys())
    db = store.select('db_bb_crop')
main_db = db
#print(main_db)
anomalous_db = main_db[main_db.Normal == False]

anomalous_db.loc[anomalous_db['bound_boxX'].map(len) == 0, 'Normal'] = True
anomalous_db = anomalous_db[anomalous_db.Normal == False]
anomalous_db['ann_sum'] = anomalous_db['orig_boxY'].map(len)

#anomalous_db.to_excel("annotations_test.xlsx")
#print()
anomalous_db = anomalous_db.reindex()
#September2021_PM8\8inch_198ch_N3311_4
#print(anomalous_db.loc[('September2021_PM8', '8inch_198ch_N3311_4')])


import numpy as np
base_dir = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db'
dir_det = 'DET/'
dir_ae = 'AE/'
images_dir_loc = 'F:/ScratchDetection/MeasurementCampaigns/'

## extract normal images for training
X_train_list_ae = np.load(os.path.join(base_dir, dir_ae, 'X_train_AE.npy'), allow_pickle=True)
X_test_list_ae = np.load(os.path.join(base_dir, dir_ae, 'X_test_AE.npy'), allow_pickle=True)

Y_train = np.load(os.path.join(base_dir, dir_det, 'Y_train_DET_final.npy'), allow_pickle=True).tolist()
Y_test = np.load(os.path.join(base_dir, dir_det, 'Y_test_DET_final.npy'), allow_pickle=True).tolist()

Y_train = np.array(Y_train).flatten()
Y_test = np.array(Y_test).flatten()
print(Y_train)
unique, counts = np.unique(Y_train, return_counts=True)
print(dict(zip(unique, counts)))
unique, counts = np.unique(Y_test, return_counts=True)
print(dict(zip(unique, counts)))


'''
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
'''
