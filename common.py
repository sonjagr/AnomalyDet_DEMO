## crop box sizes
BOXSIZE_Y = 160
BOXSIZE_X = 160
BOXSIZE = 160

## image sizes
PICTURESIZE_Y = 2748-28
PICTURESIZE_X = 3840
SS_BOX_THRESH = 20

YS = 17
XS = 24
PATCHES = XS*YS
LW = 0.85

BAD_DIR_ENDS = ('fter','DF','re','Re','IV', 'CV', 'ide', 'ment', 'sureme', 'epeat', 'ontact', 'emoval')
DEF_COLS = ["Campaign", "DUT", "Step", "FileName", "Normal"]

EIGHTBITMAX = 255
DPI = 600

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

## base directory where images are stored
imgDir_pc = 'F:/ScratchDetection/MeasurementCampaigns'
imgDir_laptop = r'/media/gsonja/Samsung_T5/ScratchDetection/MeasurementCampaigns/'
imgDir_gpu = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

## lsit of all campaigns to consider here
Campaigns = ['EndOf2021_PM8','Fall2021_PM8', 'LongTermIV_2021_ALPS', 'September2021_PM8', 'Winter2022_ALPS']
#Campaigns = ['Winter2022_ALPS']

## output file for annotations
#DataBaseFile = 'EndOf2021_PM8.h5'
DataBaseFile = 'annotation_testing_for_anomalous_br'

## directory where annotation files will be stored
DataBaseFileLocation_gpu = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
DataBaseFileLocation_local = 'db/'

TrainDir_gpu = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/'
TrainDir_pc = 'C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/'

TEST_FRAC = 0.1
VAL_FRAC = 0.1
TRAIN_FRAC = 1. - TEST_FRAC - VAL_FRAC


