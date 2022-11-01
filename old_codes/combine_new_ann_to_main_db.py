import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import os

TRAIN_DIR_LOC = '/home/gsonja/PycharmProjects/anomaly-detection-2/db'

def combine_datasets(new_db, old_db, name):
    old_db = old_db[['FileName', 'Date', 'orig_boxX', 'orig_boxY']]
    combined_db = pd.concat([old_db, new_db])
    new_date = combined_db["Date"].max()
    combined_db = combined_db[~combined_db.index.duplicated(keep='last')]
    combined_db.to_hdf(r'/home/gsonja/PycharmProjects/anomaly-detection-2/db/%s_DATABASE' % name, key='db', mode='w')
    print('New data has been added to %s_DATABASE, newest data was collected on %s' % (name, str(new_date)))
    return combined_db

def open_db(name):
    f = os.path.join(TRAIN_DIR_LOC, name)  # newest database
    with pd.HDFStore(f, mode='r') as store:
        db = store.select('db')
    return db

new_f = '/home/gsonja/PycharmProjects/anomaly-detection-2/db/pre_series'
with pd.HDFStore(new_f, mode='r') as store:
    new_db = store.select('db')
new_db = new_db.reset_index()
new_db = new_db[new_db.Campaign == 'Preseries_June2022']
new_db = new_db.set_index(['Campaign', 'DUT', 'Step'])
print(new_db)

new_train_db, new_test_val_db = train_test_split(new_db, test_size=0.2)
new_val_db, new_test_db = train_test_split(new_test_val_db, test_size=0.5)

old_train_db = open_db("TRAIN_DATABASE")
old_val_db = open_db("VAL_DATABASE")
old_test_db = open_db("TEST_DATABASE")

pickle.HIGHEST_PROTOCOL = 4
combine_datasets(new_train_db, old_train_db, "TRAIN")
combine_datasets(new_val_db, old_val_db, "VAL")
combine_datasets(new_test_db, old_test_db, "TEST")

