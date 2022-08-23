import pickle
pickle.HIGHEST_PROTOCOL = 4
import pandas as pd

'''
new_f = '/home/gsonja/PycharmProjects/anomaly-detection-2/db/extra_annotations_dummy_sensor'
with pd.HDFStore(new_f, mode='r') as store:
    new_db = store.select('db')
new_db = new_db.reset_index()
new_db['Campaign'] = 'Fake_campaign1'
new_db['DUT'] = 'Dummy_DUT1'
new_db = new_db.set_index(['Campaign', 'DUT', 'Step'])

print(new_db)

main_f = '/home/gsonja/PycharmProjects/anomaly-detection-2/db/DET/TRAIN_DATABASE_20220802'
with pd.HDFStore(main_f, mode='r') as store:
    main_db = store.select('db')

main_db.FileName = main_db.FileName.map(lambda x: x.split('\\')[-1])
print(main_db)
main_db.to_hdf('/home/gsonja/PycharmProjects/anomaly-detection-2/db/TRAIN_DATABASE_20220803', key='db', mode='w')
'''

import glob
import numpy as np

for filename in glob.glob('/media/gsonja/Samsung_T5/testing_dataset/Fake_Campaign1/Dummy_DUT05/' + '*.npy'):
    print(filename)
    npy = np.load(filename)
    print(npy.shape)
    npy_full = []
    for i in range(0,16):
        npy_full = np.append(npy_full, np.full((1,3840), 0))
    npy_full = npy_full.reshape(16, 3840)
    npy = np.concatenate((npy,npy_full), axis = 0)
    print(npy.shape)
    print(npy)
    file = filename.split('/')[-1]
    np.save('/media/gsonja/Samsung_T5/testing_dataset/Fake_Campaign1/Dummy_DUT1/'+file, npy)

