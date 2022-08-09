import pickle
pickle.HIGHEST_PROTOCOL = 4
import pandas as pd

new_f = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/xy_table_tests_Summer2021_2'
with pd.HDFStore(new_f, mode='r') as store:
    new_db = store.select('db')
new_db = new_db.reset_index()
new_db['Campaign'] = 'xy_table_tests_Summer2021'
new_db['DUT'] = 'HPK_198ch_8inch_2010'
new_db['FileName'] = new_db['FileName'].map(lambda x : x.split('\\')[-1])
new_db['Step'] = new_db['Step'].map(lambda x : x[1:])
new_db = new_db.set_index(['Campaign', 'DUT', 'Step'])

print(new_db)
main_f = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/TRAIN_DATABASE_20220805'
with pd.HDFStore(main_f, mode='r') as store:
    main_db = store.select('db')

combined_db = pd.concat([main_db, new_db])
print(main_db)
print(combined_db)

combined_db.to_hdf('C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/TRAIN_DATABASE_20220805', key='db', mode='w')
