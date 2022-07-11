import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from common import *
from helpers.annotation_helpers import match, get_bad_files_to_process, bb_to_sb
import os.path
import matplotlib.patches as patches
from matplotlib import pyplot as plt

DataBaseFile = 'extra_annotations_clean_2'
imgDir = imgDir_pc
DataBaseFileLocation = DataBaseFileLocation_local
base_dir = TrainDir_pc
dir_det = '../db/DET/'
dir_ae = '../db/AE/'
extra_cols = ["bound_boxX", "bound_boxY", "bound_box_dimX", "bound_box_dimY", "orig_boxX", "orig_boxY"]
load_from_file_list = True

loaded_db = None
## load existing database if it exists
if os.path.exists(DataBaseFileLocation + DataBaseFile):
    print("Reading existing database file from", DataBaseFile)
    with pd.HDFStore(DataBaseFileLocation + DataBaseFile, mode="r+") as store:
        loaded_db = store["db"]

## if you have a numpy array of files you would like to process, give them here, they are formatted into the database
if load_from_file_list:
    #f = os.path.join(base_dir, 'db/test_defective_df.h5')
    #with pd.HDFStore(f, mode='r') as store:
    #    train_val_db = store.select('db')
    #    print(f'Reading {DataBaseFileLocation_local + f}')
    #train_df = train_val_db
    #X_train_det_list = train_df['Path'].to_numpy()
    X_train_det_list = np.load(base_dir + dir_ae + 'X_train_forcnn_toann_2.npy')
    X_test_det_list = np.load(base_dir + dir_ae + 'X_test_forcnn_toann_2.npy')

    list_to_annotate =  X_train_det_list

    all_files_db = []
    empty_list = np.array([]).tolist()
    for i in list_to_annotate:
        _campaign, _dut, _fn = i.split('/')[-3:]
        _s = i.split('step')[-1].replace('.npy', '')
        all_files_db.append((_campaign, _dut, _s, _fn, False) + ((empty_list),) * len(extra_cols))

    all_files_db = pd.DataFrame(all_files_db, columns=DEF_COLS + extra_cols)
    all_files_db = all_files_db.set_index(keys=["Campaign", "DUT", "Step"])

    if loaded_db is None:
        loaded_db = all_files_db[all_files_db.Normal == True]
        print('No loaded database')
    else:
        loaded_db = pd.concat([loaded_db, all_files_db[all_files_db.Normal == True]], axis=0)
        loaded_db = loaded_db[~loaded_db.index.duplicated(keep="first")]

    bad_files_db = all_files_db[all_files_db.Normal == False]
    bad_files_to_process = bad_files_db[~bad_files_db.index.isin(loaded_db.index)]
    bad_files_to_process = bad_files_to_process.assign(processed=False)

## get a list of rescans from a Campaign directory
elif load_from_file_list == False:
    loaded_db, bad_files_to_process = get_bad_files_to_process(imgDir, Campaigns, extra_cols, loaded_db)

def box_size(event):
    try:
        x_end.append(int(event.xdata))
        y_end.append(int(event.ydata))
    except:
        pass

def onclick_bb(event):
    try:
        x_click = int(event.xdata)
        y_click = int(event.ydata)
        x_prev = x_end[-1]
        y_prev = y_end[-1]
        if len(y_end) % 2 != 0:
            box_ID = "%i-%i" % (x_prev, y_prev)
            update_canvas = False
            if event.dblclick:
                if match(box_ID, list(selected_boxes_bb))[0]:
                    rem = match(box_ID, list(selected_boxes_bb))[1]
                    selected_boxes_bb[rem].remove()
                    del selected_boxes_bb[rem]
                    update_canvas = True
            elif (len(list(selected_boxes_bb)) == 0 or ((match(box_ID, list(selected_boxes_bb))[0] is False))):
                dims[box_ID] = "%i-%i" % (abs(x_prev-x_click), abs(y_prev-y_click))
                selected_boxes_bb[box_ID] = patches.Rectangle(
                    (x_click-abs(x_prev-x_click), y_click- abs(y_prev-y_click)), abs(x_prev-x_click), abs(y_prev-y_click), linewidth=LW, edgecolor='r', facecolor='none', zorder = 3)
                ax.add_patch(selected_boxes_bb[box_ID])
                update_canvas = True
        if update_canvas:
            plt.draw()
            plt.pause(0.001)
    except:
        pass

def onclick_original(event):
    try:
        x_bottom = int(event.xdata/BOXSIZE_X)*BOXSIZE_X
        y_bottom = int(event.ydata/BOXSIZE_Y)*BOXSIZE_Y
        box_ID = "%i-%i" % (x_bottom, y_bottom)
        update_canvas = False
        if event.dblclick:
            if box_ID in selected_boxes_original:
                selected_boxes_original[box_ID].remove()
                del selected_boxes_original[box_ID]
                update_canvas = True
        else:
            if (box_ID not in selected_boxes_original):
                selected_boxes_original[box_ID] = patches.Rectangle(
                    (x_bottom, y_bottom), BOXSIZE_X, BOXSIZE_Y, linewidth=LW, edgecolor='r', facecolor='none', zorder = 3)
                ax.add_patch(selected_boxes_original[box_ID])
                update_canvas = True
        if update_canvas:
            plt.draw()
            plt.pause(0.001)
    except:
        pass

process_indexes = bad_files_to_process.index

print("Number of unprocessed images: ", len(process_indexes))
for _c, _img_index in enumerate(tqdm(process_indexes)):
    print(imgDir, _img_index[0], _img_index[1], bad_files_to_process.loc[_img_index, "FileName"])
    filepath = os.path.join(imgDir, _img_index[0], _img_index[1], bad_files_to_process.loc[_img_index, "FileName"])
    if _img_index[0] == 'September2021_PM8' and _img_index[1] == '8inch_198ch_N3311_7':
        filepath = os.path.join(imgDir, _img_index[0], _img_index[1] + '/before', bad_files_to_process.loc[_img_index, "FileName"])
    selected_boxes_bb, dims = {}, {}
    x_end, y_end = [], []
    fig, ax = plt.subplots(figsize=(16, 10))
    cid = fig.canvas.mpl_connect('button_press_event', onclick_bb)
    bsz = fig.canvas.mpl_connect('button_press_event', box_size)
    im = np.load(filepath)
    im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
    ax.imshow(im[0:PICTURESIZE_Y, 0:PICTURESIZE_X])
    ax.set_xticks(np.arange(80, PICTURESIZE_X, BOXSIZE_X), minor = True)
    ax.set_yticks(np.arange(80, PICTURESIZE_Y, BOXSIZE_Y), minor = True)
    ax.set_xticks(np.arange(0, PICTURESIZE_X, BOXSIZE_X))
    ax.set_yticks(np.arange(0, PICTURESIZE_Y, BOXSIZE_Y))
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.tick_params(axis='x', which='major', labelsize=16, labelrotation=90)
    #ax.grid(which = 'minor', color='w', linewidth=0.3, linestyle ='-.', zorder = -2)
    #ax.grid(which= 'major', color='w', linewidth=0.4, linestyle='--', zorder = -2)
    plt.xlabel('x', fontsize =16)
    plt.ylabel('y', fontsize =16)
    plt.title(str(_img_index),  fontsize =16)
    plt.show()
    plt.pause(0.01)

    bbsX, bbsY = [], []
    for box_ID in selected_boxes_bb:
        b_x, b_y = tuple(box_ID.split("-"))
        bbsX.append(b_x)
        bbsY.append(b_y)

    dims = {box: dims[box] for box in list(selected_boxes_bb)}
    dimsX, dimsY = [], []
    for box_ID in dims:
        d_x, d_y = tuple(dims.get(box_ID).split("-"))
        dimsX.append(d_x)
        dimsY.append(d_y)

    bad_files_to_process.at[_img_index, "bound_boxX"] = bbsX
    bad_files_to_process.at[_img_index, "bound_boxY"] = bbsY

    bad_files_to_process.at[_img_index, "bound_box_dimX"] = dimsX
    bad_files_to_process.at[_img_index, "bound_box_dimY"] = dimsY

    selected_boxes_original = bb_to_sb(bbsX, bbsY, dimsX, dimsY)
    olb = len(list(selected_boxes_original))
    print('Number of overlapping sub-boxes: ', olb)

    if olb > 0:
        fig, ax = plt.subplots(figsize=(16, 10))
        cid = fig.canvas.mpl_connect('button_press_event', onclick_original)
        im = np.load(filepath)
        im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
        ax.imshow(im[0:PICTURESIZE_Y, 0:PICTURESIZE_X])
        for k in range(0, len(bbsX)):
            print(abs(int(dimsX[k])), abs(int(dimsY[k])))
            ax.add_patch(patches.Rectangle((int(bbsX[k]), int(bbsY[k])), (int(dimsX[k])), (int(dimsY[k])), linewidth=LW, edgecolor='y', facecolor='none', zorder = 3))
        for box_ID in list(selected_boxes_original):
            ax.add_patch(selected_boxes_original[box_ID])
        ax.set_xticks(np.arange(0, PICTURESIZE_X, BOXSIZE_X))
        ax.set_yticks(np.arange(0, PICTURESIZE_Y, BOXSIZE_Y))
        ax.tick_params(axis='y', which='major', labelsize=16)
        ax.tick_params(axis='x', which='major', labelsize=16, labelrotation = 90)
        ax.grid(which='major',  color='w', linewidth=0.4, linestyle='--', zorder = -2)
        plt.title(str(_img_index),  fontsize =16)
        plt.xlabel('x',  fontsize =16)
        plt.ylabel('y', fontsize =16)
        plt.show()
        plt.pause(0.001)

    boxesX, boxesY = [], []
    for box_ID in selected_boxes_original:
        b_x, b_y = tuple(box_ID.split("-"))
        boxesX.append(b_x)
        boxesY.append(b_y)

    bad_files_to_process.at[_img_index, "orig_boxX"] = boxesX
    bad_files_to_process.at[_img_index, "orig_boxY"] = boxesY

    procd = True
    end = input(" Type 'x' to exit program, 's' to ignore previous annotation. Press enter to continue:  ")
    if "s" in end:
        print('Skip: previous image skipped')
        procd = False

    bad_files_to_process.at[_img_index, "processed"] = procd

    if "x" in end:
        print('Quit: annotations will be saved')
        break

#6. Determine which files were processed and can be joined with the database
bad_files_processed = bad_files_to_process[bad_files_to_process.processed==True]
bad_files_processed = bad_files_processed.drop(columns=["processed"])
bad_files_processed['Date'] = pd.to_datetime('today').date()

loaded_db = pd.concat([loaded_db, bad_files_processed], axis = 0)

#7.: write out database
DataBaseFile = 'extra_annotations_clean_2'
loaded_db.to_hdf(DataBaseFileLocation + DataBaseFile, key='db', mode='w')