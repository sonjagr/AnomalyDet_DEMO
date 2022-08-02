import pickle
pickle.HIGHEST_PROTOCOL = 4
import pandas as pd

new_f = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/extra_annotations_dummy_sensor'
with pd.HDFStore(new_f, mode='r') as store:
    new_db = store.select('db')
new_db = new_db.reset_index()
new_db['Campaign'] = 'Fake_campaign1'
new_db['DUT'] = 'Dummy_DUT1'
new_db = new_db.set_index(['Campaign', 'DUT', 'Step'])

print(new_db)
main_f = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/TRAIN_DATABASE_20220711'
with pd.HDFStore(main_f, mode='r') as store:
    main_db = store.select('db')

combined_db = pd.concat([main_db, new_db])
print(main_db)
print(combined_db)

combined_db.to_hdf('C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/TRAIN_DATABASE_20220802', key='db', mode='w')
