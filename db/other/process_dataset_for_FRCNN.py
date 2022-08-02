import pandas as pd
import os
import numpy as np
from common import *
from helpers.dataset_helpers import box_to_labels
from sklearn.model_selection import train_test_split

"Script that reads in the annotated files and creates separate files containing the image files for anomalous and non-anomalous images"

np.random.seed(42)
f = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/three_annotations/main_db_bb_crop.h5'
with pd.HDFStore( f,  mode='r') as store:
        db = store.select('db_bb_crop')
        print(f'Reading {DataBaseFileLocation_local+f}')
main_db = db

## select only normal, ie not anomalous images
##normal_db = main_db[main_db.Normal == True]
anomalous_db = main_db[main_db.Normal == False]

##normal_db = normal_db.reset_index()
anomalous_db = anomalous_db.reset_index()

## for now, we are dropping one DUT because the path is unusual
##normal_db = normal_db.drop(normal_db.loc[(normal_db.Campaign == 'September2021_PM8') & (normal_db.DUT == '8inch_198ch_N3311_7')].index)
##normal_db['Path'] = normal_db.Campaign + '/' + normal_db.DUT + '/' + normal_db.FileName
##normal_files = normal_db.Path
##print('Number of normal files: ', len(normal_files))

anomalous_db = anomalous_db.drop(anomalous_db.loc[(anomalous_db.Campaign == 'September2021_PM8') & (anomalous_db.DUT == '8inch_198ch_N3311_7')].index)
anomalous_db['Path'] = anomalous_db.Campaign + '/' + anomalous_db.DUT + '/' + anomalous_db.FileName
anomalous_files = anomalous_db.Path.to_numpy()
print('Number of anomalous files: ', len(anomalous_files))

x1 = anomalous_db.bound_boxX.to_numpy()
y1 = anomalous_db.bound_boxY.to_numpy()

plus_x2 = anomalous_db.bound_box_dimX.to_numpy()
plus_y2 = anomalous_db.bound_box_dimY.to_numpy()

x2, y2 = [], []
for i in range(0,len(x1)):
        x2_elem, y2_elem = [], []
        for j in range(0, len(x1[i])):
                x2_elem.append(int(x1[i][j]) + int(plus_x2[i][j]))
                y2_elem.append(int(y1[i][j]) + int(plus_y2[i][j]))
        x2.append(x2_elem)
        y2.append(y2_elem)

filenames = []
labels = []
boxx1, boxx2, boxy1, boxy2 = [], [], [], []
boxes = []
print(len(x2), len(y2))
myfile = open('/db/other/bbs_for_YOLO.txt', 'w')
for k in range(0, len(x2)):
        boxlist = []
        filenames.append(anomalous_files[k])
        for h in range(0, len(x1[k])):
                boxlist.append(x1[k][h])
                boxlist.append(y1[k][h])
                boxlist.append(x2[k][h])
                boxlist.append(y2[k][h])
                boxlist.append(1)
        boxes.append(boxlist)
        if len(boxlist)>0:
                myfile.write('F:/ScratchDetection/MeasurementCampaigns/'+str(anomalous_files[k][:-3])+'jpeg'+' ')
                num = 1
                for i in boxlist:
                        if num %5 != 0:
                                myfile.write(str(i)+',')
                        if num %5 == 0:
                                myfile.write(str(i)+' ')
                        num = num + 1
                myfile.write('\n')
myfile.close
print(len(boxes), len(filenames))
data_loc = 'F:/ScratchDetection/MeasurementCampaigns/'
filenames = [w.replace('.npy', '.jpeg') for w in filenames]
filenames = [data_loc + s for s in filenames]

csv_df = pd.DataFrame(list(zip(filenames, boxes)), columns =['a','b'])

#csv_df.to_csv(path_or_buf='C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/db/bbs_for_YOLO.csv', sep=',', na_rep='', float_format=None,  header=False, index=False)