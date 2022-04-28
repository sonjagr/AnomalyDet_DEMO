import os, sys, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from common import *
import os.path, sys
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

imgDir = imgDir_local
DataBaseFileLocation = DataBaseFileLocation_local

normal_image_names = "^initial_scan_pad(?P<Pad>.*)_step(?P<Step>.*).npy"
anomalous_image_names = "^rescan_pad(?P<Pad>.*)_step(?P<Step>.*).npy"
npy_file = ".npy"

rescan_file_exp = re.compile(anomalous_image_names)
initial_file_exp = re.compile(normal_image_names)
npy_file_exp = re.compile(npy_file)

if not os.path.exists(imgDir):
    sys.exit(imgDir+" as directory where input files are supposed to be does not exist.")
CampaignsInDirectory = [_dir for _dir in os.listdir(imgDir) if os.path.isdir(os.path.join(imgDir, _dir))]

if len(Campaigns)==1 and Campaigns[0] == "all":
    Campaigns = os.listdir(imgDir)

#1. this loop gets a list of all potential images with defects
all_files_db = []
empty_list = np.array([]).tolist()
# loop over the campaigns
for _campaign in Campaigns:
    _campaignpath = os.path.join(imgDir, _campaign)
    # loop over the duts
    for _dut in os.listdir(_campaignpath):
        if _dut.endswith(('0','1','2','3','4','5','6','7','8','9')):
            _subdirpath = os.path.join(_campaignpath, _dut)
            if not os.path.isdir(_subdirpath):
                continue
            print("Reading",_subdirpath)
            dut_good_pad_images = {}
            dut_bad_pad_images = {}
            # loop over the single images and separate initial and re scans
            for _file in os.listdir(_subdirpath):
                match_rescan = re.search(rescan_file_exp, _file)
                match_initialscan = re.search(initial_file_exp, _file)
                if match_rescan is not None:
                    dut_bad_pad_images[int(match_rescan["Step"])] = _file
                elif match_initialscan is not None:
                    dut_good_pad_images[int(match_initialscan["Step"])] = _file
            #if same index is in both, overwrite the rescans by the bad and delete the copy from the list of good sensors
            for bad_pad_index in dut_bad_pad_images:
                dut_bad_pad_images[bad_pad_index] = dut_good_pad_images[bad_pad_index]
                del dut_good_pad_images[bad_pad_index]
            print('Number of rescans: ', len(dut_bad_pad_images))
            for _s in sorted(dut_good_pad_images):
                all_files_db.append((_campaign, _dut, _s, dut_good_pad_images[_s], True, empty_list, empty_list))
            for _s in sorted(dut_bad_pad_images):
                all_files_db.append((_campaign, _dut, _s, dut_bad_pad_images[_s], False, empty_list, empty_list))

#this is a pandas data frame containing all images read and sorted from the ImageDir
all_files_db = pd.DataFrame(all_files_db, columns=["Campaign", "DUT", "Step", "FileName", "Normal", "BoxX", "BoxY"])
all_files_db = all_files_db.set_index(keys=["Campaign", "DUT", "Step"])

#2. load existing database file that is or is not empty
loaded_db = None
if os.path.exists(DataBaseFileLocation + DataBaseFile):
    print("Reading existing database file from", DataBaseFile)
    with pd.HDFStore(DataBaseFileLocation + DataBaseFile, mode="r+") as store:
        loaded_db = store["db"]

#3. either create database or extend excisting database with the good images
if loaded_db is None:
    loaded_db = all_files_db[all_files_db.Normal==True]
else:
    loaded_db = pd.concat([loaded_db, all_files_db[all_files_db.Normal==True]], axis=0)
    loaded_db = loaded_db[~loaded_db.index.duplicated(keep="first")]

#4. get list of all bad pads which are not part of the database
bad_files_db = all_files_db[all_files_db.Normal==False]
bad_files_loaded_db = loaded_db[loaded_db.Normal==False]
bad_files_to_process = bad_files_db[~bad_files_db.index.isin(loaded_db.index)]
bad_files_to_process = bad_files_to_process.assign(processed=False)

#5. loop over all files to process and perform the selection
def onclick(event):
    try:
        x_bottom = int(event.xdata/BOXSIZE_X)*BOXSIZE_X
        y_bottom = int(event.ydata/BOXSIZE_Y)*BOXSIZE_Y
        box_ID = "%i-%i" % (x_bottom, y_bottom)
        update_canvas = False
        if event.dblclick:
            if box_ID in selected_boxes:
                selected_boxes[box_ID].remove()
                del selected_boxes[box_ID]
                update_canvas = True
        else:
            if not box_ID in selected_boxes:
                selected_boxes[box_ID] = patches.Rectangle(
                    (x_bottom, y_bottom), BOXSIZE_X, BOXSIZE_Y, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(selected_boxes[box_ID])
                update_canvas = True
        if update_canvas:
            plt.draw()
            plt.pause(0.001)
    except:
        pass

process_indexes = bad_files_to_process.index
print(len(process_indexes), "images to annotate")
skipped_files = []
for _c, _img_index in enumerate(tqdm(process_indexes)):
    if _c > 0:
        end = input(" Type 'x' to exit program, enter to continue")
        if end == "x":
            break
    filepath = os.path.join(imgDir, _img_index[0], _img_index[1], bad_files_to_process.loc[_img_index, "FileName"])
    selected_boxes = {}
    fig, ax = plt.subplots(figsize=(20, 12))
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    im = np.load(filepath)
    im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
    ax.imshow(im[0:PICTURESIZE_Y, 0:PICTURESIZE_X])
    #plt.title('Select all areas containing anomalies (dust, scratch, stain, ...) by clicking. \n You can un-select by double-clicking. \n When done close window.', fontsize = 14)
    plt.title(str(_img_index))
    plt.show()
    plt.pause(0.001)
    boxesX, boxesY = [], []
    for box_ID in selected_boxes:
        b_x, b_y = tuple(box_ID.split("-"))
        boxesX.append(b_x)
        boxesY.append(b_y)
    bad_files_to_process.at[_img_index, "BoxX"] = boxesX
    bad_files_to_process.at[_img_index, "BoxY"] = boxesY
    bad_files_to_process.at[_img_index, "processed"] = True

#6. Determine which files were processed and can be joined with the database
bad_files_processed = bad_files_to_process[bad_files_to_process.processed==True]
bad_files_processed = bad_files_processed.drop(columns=["processed"])

loaded_db = pd.concat([loaded_db, bad_files_processed])

#7.: write out database
with pd.HDFStore(DataBaseFileLocation + DataBaseFile, mode="w") as store:
    store["db"] = loaded_db