import pandas as pd
import os
import numpy as np
from common import *
from helpers.dataset_helpers import box_to_labels
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"Script that reads in the annotated files and creates separate files containing the image files for anomalous and non-anomalous images"

def save_file(path, nplist, overwrite):
    if overwrite:
        np.save(path, nplist)
        print('Saving...', path)
    else:
        if not os.path.exists(path):
            np.save(path, nplist)
            print('Saving...', path)

np.random.seed(42)
f = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/main_db_bb_crop.h5'
with pd.HDFStore( f,  mode='r') as store:
        db = store.select('db')
        print(f'Reading {DataBaseFileLocation_local+f}')
main_db = db

np.random.seed(42)
f = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/annotation_testing_for_anomalous_br'
with pd.HDFStore( f,  mode='r') as store:
        db = store.select('db')
        print(f'Reading {DataBaseFileLocation_local+f}')
backround_db = db[db.Normal == False][['FileName', 'orig_boxX', 'orig_boxY']].reset_index()
anomalous_backround_db = backround_db.rename(columns={"orig_boxX": "orig_boxX_br", "orig_boxY": "orig_boxY_br"})
print(anomalous_backround_db.head())

## select only normal, ie not anomalous images
normal_db = main_db[main_db.Normal == True].reset_index()
anomalous_db = main_db[main_db.Normal == False].reset_index()

## for now, we are dropping one DUT because the path is unusual
normal_db = normal_db.drop(normal_db.loc[(normal_db.Campaign == 'September2021_PM8') & (normal_db.DUT == '8inch_198ch_N3311_7')].index)
normal_db['Path'] = normal_db.Campaign + '/' + normal_db.DUT + '/' + normal_db.FileName
normal_files = normal_db.Path
print('Number of normal files: ', len(normal_files))

anomalous_db = anomalous_db.drop(anomalous_db.loc[(anomalous_db.Campaign == 'September2021_PM8') & (anomalous_db.DUT == '8inch_198ch_N3311_7')].index)
anomalous_db['Path'] = anomalous_db.Campaign + '/' + anomalous_db.DUT + '/' + anomalous_db.FileName
anomalous_backround_db = anomalous_backround_db.drop(anomalous_backround_db.loc[(anomalous_backround_db.Campaign == 'September2021_PM8') & (anomalous_backround_db.DUT == '8inch_198ch_N3311_7')].index)
anomalous_backround_db['Path'] = anomalous_backround_db.Campaign + '/' + anomalous_backround_db.DUT + '/' + anomalous_backround_db.FileName
#join from path
anomalous_backround_db = anomalous_backround_db[['Path', 'orig_boxX_br', 'orig_boxY_br']]
anomalous_db = anomalous_db[['Path', 'orig_boxX', 'orig_boxY']]
anomalous_db_comb = anomalous_db.merge(anomalous_backround_db, on='Path', how='inner')
print(anomalous_db_comb.columns)
anomalous_files = anomalous_db_comb.Path
print('Number of anomalous files: ', len(anomalous_files))

anomalous_labels = anomalous_db_comb[['orig_boxX_br', 'orig_boxY_br', 'orig_boxX', 'orig_boxY']].copy()
anomalous_labels['crop_lbls'] = pd.NaT
anomalous_labels['crop_lbls'] = anomalous_labels.apply(lambda x: box_to_labels(x.orig_boxX, x.orig_boxY), axis=1)
anomalous_labels['crop_lbls_br'] = pd.NaT
anomalous_labels['crop_lbls_br'] = anomalous_labels.apply(lambda x: box_to_labels(x.orig_boxX_br, x.orig_boxY_br), axis=1)

## split into training and test sets
X_train_normal_list, X_test_normal_list = train_test_split(normal_files.values, test_size=0.2, shuffle=True, random_state=42)
X_train_anomalous_list, X_test_anomalous_list, Y_train_anomalous_list, Y_test_anomalous_list = train_test_split(anomalous_files.values, anomalous_labels.crop_lbls.values, test_size=0.2, shuffle=True, random_state=2)
X_train_anomalous_list_br, X_test_anomalous_list_br, Y_train_anomalous_list_br, Y_test_anomalous_list_br = train_test_split(anomalous_files.values, anomalous_labels.crop_lbls_br.values, test_size=0.2, shuffle=True, random_state=2)

X_train_backround = X_train_anomalous_list_br
Y_train_backround = Y_train_anomalous_list_br

X_test_backround = X_test_anomalous_list_br
Y_test_backround = Y_test_anomalous_list_br


#### SAVING....

base_dir = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/'
dir_ae = "AE/"
dir_det = "DET/"
dirs_ae = os.listdir(base_dir + dir_ae)
dirs_det = os.listdir(base_dir + dir_det)

np.random.seed(1)
X_train_AE = np.random.choice(X_train_normal_list, 16000, replace=False)
X_test_AE = np.random.choice(X_test_normal_list, 4000, replace=False)

#save_file(base_dir + dir_ae + 'X_train_AE.npy', X_train_AE, False)
#save_file(base_dir + dir_ae + 'X_test_AE.npy', X_test_AE, False)

X_train_list_normal_removed = [x for x in X_train_normal_list if x not in X_train_AE]
X_test_list_normal_removed = [x for x in X_test_normal_list if x not in X_test_AE]

#save_file(base_dir + dir_ae + 'X_train_NOT_AE.npy', X_train_list_normal_removed, True)
#save_file(base_dir + dir_ae + 'X_test_NOT_AE.npy', X_test_list_normal_removed, True)

save_file(base_dir + dir_det + 'X_train_DET_final.npy', X_train_anomalous_list,  False)
save_file(base_dir + dir_det + 'X_test_DET_final.npy', X_test_anomalous_list,  False)

save_file(base_dir + dir_det + 'Y_train_DET_final.npy', Y_train_anomalous_list,  False)
save_file(base_dir + dir_det + 'Y_test_DET_final.npy', Y_test_anomalous_list, False)

print(f'Number of training samples for AE is {len(X_train_AE)} and testing samples {len(X_test_AE)}')
print(f'Number of training samples not used for AE is {len(X_train_list_normal_removed)} and testing samples {len(X_test_list_normal_removed)}')
print(f'Number of training samples for CNN is {len(X_train_anomalous_list)} and testing samples {len(X_test_anomalous_list)}')
print(f'Number of training samples for backround is {len( X_train_backround)} and testing samples {len(X_test_backround)}')

save_file(base_dir + dir_det + 'X_train_DET_backround.npy', X_train_backround,  False)
save_file(base_dir + dir_det + 'X_test_DET_backround.npy', X_test_backround, False)

save_file(base_dir + dir_det + 'Y_train_DET_backround.npy', Y_train_backround,  False)
save_file(base_dir + dir_det + 'Y_test_DET_backround.npy', Y_test_backround, False)


