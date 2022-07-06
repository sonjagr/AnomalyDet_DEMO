import pickle
pickle.HIGHEST_PROTOCOL = 4
import pandas as pd
import os
import numpy as np
from common import *
from helpers.dataset_helpers import box_to_labels
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
orig_db = db[db.Normal == False]

print(orig_db)

np.random.seed(42)
f = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/extra_annotaions_from_normals'
with pd.HDFStore( f,  mode='r') as store:
        db = store.select('db')
        print(f'Reading {DataBaseFileLocation_local+f}')
extra_db = db

print(extra_db)

whole_db = pd.concat([orig_db, extra_db])
whole_db = whole_db[whole_db['orig_boxY'].map(len) > 0]
whole_db = whole_db.reset_index()[['Campaign', 'DUT', 'FileName', 'orig_boxY', 'orig_boxX']]
whole_db = whole_db.drop(whole_db.loc[(whole_db.Campaign == 'September2021_PM8') & (whole_db.DUT == '8inch_198ch_N3311_7')].index)
whole_db['Path'] = whole_db.Campaign + '/' + whole_db.DUT + '/' + whole_db.FileName
whole_db = whole_db.drop(['Campaign', 'DUT', 'FileName'], axis =1)
whole_db['crop_lbls'] = pd.NaT
whole_db['crop_lbls'] = whole_db.apply(lambda x: box_to_labels(x.orig_boxX, x.orig_boxY).flatten(), axis=1)
whole_db['Date'] = pd.to_datetime('today').date()
print(whole_db)

train_df = whole_db.sample(frac=0.9)
test_df = whole_db.drop(train_df.index)

train_imgs = train_df['Path'].to_numpy()
train_lbls = train_df['crop_lbls'].to_numpy()

print(train_df)
print(len(train_lbls))

train_df.to_hdf('C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/train_defective_df.h5', key='db', mode='w')
test_df.to_hdf('C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/test_defective_df.h5', key='db', mode='w')



