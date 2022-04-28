## CROP BOX SIZES
BOXSIZE_Y = 160
BOXSIZE_X = 160

## IMAGES MUST BE CROPPED IN Y DIRECTION TO ALLOW BOX SIZES
PICTURESIZE_Y = 2748-28
PICTURESIZE_X = 3840
SS_BOX_THRESH = 20

YS = 17
XS = 24
LW = 0.85

BAD_DIR_ENDS = ('fter','DF','re','Re','IV', 'CV', 'ide', 'ment', 'sureme', 'epeat', 'ontact', 'emoval')
DEF_COLS = ["Campaign", "DUT", "Step", "FileName", "Normal"]

EIGHTBITMAX = 255

ANCHOR_GRID = []
ANCHOR_GRID_STR = []
for row in range(int(BOXSIZE_Y/2), PICTURESIZE_Y+int(BOXSIZE_Y/2), BOXSIZE_Y):
    for col in range(int(BOXSIZE_X/2), PICTURESIZE_X+int(BOXSIZE_Y/2), BOXSIZE_X):
        ANCHOR_GRID.append([col, row])
        ANCHOR_GRID_STR.append( "%i-%i" % (col, row))

ANCHOR_GRID_OFFSET_X = []
for row in range(int(BOXSIZE_Y/2), PICTURESIZE_Y+int(BOXSIZE_Y/2), BOXSIZE_Y):
    for col in range(int(BOXSIZE_X/2)+int(BOXSIZE_Y/2), PICTURESIZE_X, BOXSIZE_X):
        ANCHOR_GRID_OFFSET_X.append([col, row])

ANCHOR_GRID_OFFSET_X_Y = []
for row in range(int(BOXSIZE_Y/2), PICTURESIZE_Y, int(BOXSIZE_X/2)):
    for col in range(int(BOXSIZE_X/2), PICTURESIZE_X, int(BOXSIZE_X/2)):
        ANCHOR_GRID_OFFSET_X_Y.append([col, row])

## BASE DIRECTORY WHERE IMAGES ARE STORED
imgDir_local = r'F:/ScratchDetection/MeasurementCampaigns/'
imgDir_laptop_local = r'/media/gsonja/Samsung_T5/ScratchDetection/MeasurementCampaigns/'
imgDir_gpu = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

## LIST OF CAMPAIGNS HERE
Campaigns = ['EndOf2021_PM8','Fall2021_PM8', 'LongTermIV_2021_ALPS', 'September2021_PM8', 'Winter2022_ALPS']
Campaigns = ['Fall2021_PM8']

## ANNOTATED IMAGES OUTPUT FILE
#DataBaseFile = 'EndOf2021_PM8.h5'
DataBaseFile = 'annotation_testing_delete'

## DIRECTORY WHERE DF CONTAINING ANNOTATIONS WILL BE STORED
DataBaseFileLocation_gpu = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/'
DataBaseFileLocation_local = 'db/three_annotations/'

TrainDir_gpu = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/'

TEST_FRAC = 0.2
VAL_FRAC = 0.1
TRAIN_FRAC = 1. - TEST_FRAC - VAL_FRAC


