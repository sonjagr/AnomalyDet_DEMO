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
f = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/test_annotations_cleaned'
with pd.HDFStore( f,  mode='r') as store:
        db = store.select('db')
        print(f'Reading {DataBaseFileLocation_local+f}')
db.to_hdf('C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/test_annotations_cleaned', key='db', mode='w')

'''
def process_anomalous_df_to_numpy(whole_db):
    whole_db = whole_db[whole_db['orig_boxY'].map(len) > 0]
    whole_db = whole_db.reset_index()[['Campaign', 'DUT', 'FileName', 'orig_boxY', 'orig_boxX']]
    whole_db = whole_db.drop(whole_db.loc[(whole_db.Campaign == 'September2021_PM8') & (whole_db.DUT == '8inch_198ch_N3311_7')].index)
    whole_db['Path'] = whole_db.Campaign + '/' + whole_db.DUT + '/' + whole_db.FileName
    whole_db = whole_db.drop(['Campaign', 'DUT', 'FileName'], axis =1)
    whole_db['crop_lbls'] = pd.NaT
    whole_db['crop_lbls'] = whole_db.apply(lambda x: box_to_labels(x.orig_boxX, x.orig_boxY).flatten(), axis=1)
    X_list = whole_db['Path'].to_numpy()
    Y_list = whole_db['crop_lbls'].to_numpy().flatten()

whole_db.to_hdf('C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/test_defective_df.h5', key='db', mode='w')
#test_df.to_hdf('C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/test_defective_df.h5', key='db', mode='w')
'''


