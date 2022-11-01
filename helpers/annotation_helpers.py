from common import *
import os, re, sys
import pandas as pd
import numpy as np

## calculates the coordinates of orig box in required format, (x1, y1, x2, y2), upper left and lower right
def orig_box_to_coord(x,y):
    x1, y1 = int(x), int(y)
    return (x1, y1, x1+BOXSIZE_X, y1+BOXSIZE_Y)

## calculates the coordinates of bounding box in required format, (x1, y1, x2, y2)
def bbox_to_coord(x,y, dx, dy):
    x1, y1, dx, dy = int(x), int(y), int(dx), int(dy)
    return (x1, y1, x1+dx, y1+dy)

## calculates if two rectanges intersect, namely bounding boxes and original boxes, returns intersection
def intersection(a, b):
    x1 = max(min(a[0], a[2]), min(b[0], b[2]))
    y1 = max(min(a[1], a[3]), min(b[1], b[3]))
    x2 = min(max(a[0], a[2]), max(b[0], b[2]))
    y2 = min(max(a[1], a[3]), max(b[1], b[3]))
    if x1 < x2 and y1 < y2:
        return (x1, y1, x2, y2)

## determine if (clicked) point is inside rectange of original box size
def pointInRect(point,rect):
    x1, y1= rect
    w,h = BOXSIZE_X, BOXSIZE_Y
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False

## determine if box_id matches a box in list, assumes same size
## this should be updated to work with any size boxes
def match(box_id, boxes):
    c_x, c_y = tuple(box_id.split("-"))
    c_x, c_y = int(c_x)+(BOXSIZE_X/2), int(c_y)+(BOXSIZE_Y/2)
    for k in range(0,len(boxes)):
        b_x, b_y =  tuple(boxes[k].split("-"))
        b_x, b_y = int(b_x), int(b_y)
        if pointInRect((c_x, c_y), (b_x,b_y)):
            return (True, boxes[k])
    return (False, None)

# if two rectangles given in format (x1, y1, x2, y2) overlap
def overlap_f(a, b):
    if (a[1] >= b[2] or a[0] >= b[3] or a[2] <= b[1] or a[3] <= b[0]):
        return False
    return True

## function to convert bounding boxes to single shot boxes
def bb_to_sb(bbsX, bbsY, dimsX, dimsY):
    import matplotlib.patches as patches
    anchors = ANCHOR_GRID
    selected_boxes_ol = {}
    for i in range(0,len(bbsX)):
        rr1 = [int(bbsX[i]), int(bbsY[i]),int(bbsX[i])+int(dimsX[i]), int(bbsY[i])+int(dimsY[i])]
        for j in range(0,int(len(anchors))):
            rr2 = [anchors[j][1]-BOXSIZE_X/2, anchors[j][0]-BOXSIZE_X/2, anchors[j][1] + BOXSIZE_X/2, anchors[j][0] + BOXSIZE_X/2]
            if overlap_f(rr1, rr2) == True:
                box_ID  = "%i-%i" % (anchors[j][0]-BOXSIZE_X/2, anchors[j][1]-BOXSIZE_X/2)
                selected_boxes_ol[box_ID] = patches.Rectangle(
                    (anchors[j][0]-BOXSIZE_X/2, anchors[j][1]-BOXSIZE_X/2), BOXSIZE_X, BOXSIZE_Y, linewidth=LW, edgecolor='r', facecolor='none', zorder = 3)
    return selected_boxes_ol

## calculates the closest node to a given node and outputs also the distance (point-node)
from scipy.spatial import distance
def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    cn = nodes[closest_index]
    dx = node[0] - cn[0]
    dy = node[1] - cn[1]
    return cn[0], cn[1], dx, dy

## gets list of bad files (rescans) that are not in database yet (TQ)
def get_bad_files_to_process(imgDir, Campaigns, extra_cols, loaded_db):
    normal_image_names = "^initial_scan_pad(?P<Pad>.*)_step(?P<Step>.*).npy"
    anomalous_image_names = "^rescan_pad(?P<Pad>.*)_step(?P<Step>.*).npy"

    rescan_file_exp = re.compile(anomalous_image_names)
    initial_file_exp = re.compile(normal_image_names)

    if not os.path.exists(imgDir):
        sys.exit(imgDir + " as directory where input files are supposed to be does not exist.")
    CampaignsInDirectory = [_dir for _dir in os.listdir(imgDir) if os.path.isdir(os.path.join(imgDir, _dir))]

    if len(Campaigns) == 1 and Campaigns[0] == "all":
        Campaigns = os.listdir(imgDir)

    all_files_db = []
    empty_list = np.array([]).tolist()
    for _campaign in Campaigns:
        _campaignpath = os.path.join(imgDir, _campaign)
        # loop over the duts
        for _dut in os.listdir(_campaignpath):
            if not _dut.endswith(BAD_DIR_ENDS):
                _subdirpath = os.path.join(_campaignpath, _dut)
                if not os.path.isdir(_subdirpath):
                    continue
                #print("Reading", _subdirpath)
                if _subdirpath == 'F:/ScratchDetection/MeasurementCampaigns/September2021_PM8\8inch_198ch_N3311_7':
                    _subdirpath = _subdirpath + '/before'
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
                for bad_pad_index in dut_bad_pad_images:
                    dut_bad_pad_images[bad_pad_index] = dut_good_pad_images[bad_pad_index]
                    del dut_good_pad_images[bad_pad_index]
                #print('Total number of rescans: ', len(dut_bad_pad_images))
                for _s in sorted(dut_good_pad_images):
                    all_files_db.append(
                        (_campaign, _dut, _s, dut_good_pad_images[_s], True) + ((empty_list),) * len(extra_cols))
                for _s in sorted(dut_bad_pad_images):
                    all_files_db.append(
                        (_campaign, _dut, _s, dut_bad_pad_images[_s], False) + ((empty_list),) * len(extra_cols))

    # this is a pandas data frame containing all images read and sorted from the ImageDir with the info of rescanned or not
    all_files_db = pd.DataFrame(all_files_db, columns=DEF_COLS + extra_cols)
    all_files_db = all_files_db.set_index(keys=["Campaign", "DUT", "Step"])

    # 3. either create database or extend existing database with the good images
    if loaded_db is None:
        loaded_db = all_files_db[all_files_db.Normal == True]
    else:
        loaded_db = pd.concat([loaded_db, all_files_db[all_files_db.Normal == True]], axis=0)
        loaded_db = loaded_db[~loaded_db.index.duplicated(keep="first")]

    # 4. get list of all bad pads which are not part of the database
    bad_files_db = all_files_db[all_files_db.Normal == False]
    ## bad files that are in the new database but not in loaded
    bad_files_to_process = bad_files_db[~bad_files_db.index.isin(loaded_db.index)]
    bad_files_to_process = bad_files_to_process.assign(processed=False)
    return loaded_db, bad_files_to_process


def get_all_files_to_process(DataBaseFileLocation, DataBaseFile, imgDir, Campaigns, extra_cols):
    normal_image_names = "^initial_scan_pad(?P<Pad>.*)_step(?P<Step>.*).npy"
    anomalous_image_names = "^rescan_pad(?P<Pad>.*)_step(?P<Step>.*).npy"

    rescan_file_exp = re.compile(anomalous_image_names)
    initial_file_exp = re.compile(normal_image_names)

    if not os.path.exists(imgDir):
        sys.exit(imgDir + " as directory where input files are supposed to be does not exist.")
    CampaignsInDirectory = [_dir for _dir in os.listdir(imgDir) if os.path.isdir(os.path.join(imgDir, _dir))]

    if len(Campaigns) == 1 and Campaigns[0] == "all":
        Campaigns = os.listdir(imgDir)

    all_files_db = []
    empty_list = np.array([]).tolist()
    for _campaign in Campaigns:
        _campaignpath = os.path.join(imgDir, _campaign)
        # loop over the duts
        for _dut in os.listdir(_campaignpath):
            if not _dut.endswith(BAD_DIR_ENDS):
                _subdirpath = os.path.join(_campaignpath, _dut)
                if not os.path.isdir(_subdirpath):
                    continue
                print("Reading", _subdirpath)
                if _subdirpath == 'F:/ScratchDetection/MeasurementCampaigns/September2021_PM8\8inch_198ch_N3311_7':
                    _subdirpath = _subdirpath + '/before'
                dut_good_pad_images = {}
                # loop over the single images and separate initial and re scans
                for _file in os.listdir(_subdirpath):
                    match_initialscan = re.search(initial_file_exp, _file)
                    if match_initialscan is not None:
                        dut_good_pad_images[int(match_initialscan["Step"])] = _file
                for _s in sorted(dut_good_pad_images):
                    all_files_db.append(
                        (_campaign, _dut, _s, dut_good_pad_images[_s], True) + ((empty_list),) * len(extra_cols))

    # this is a pandas data frame containing all images read and sorted from the ImageDir with the info of rescanned or not
    all_files_db = pd.DataFrame(all_files_db, columns=DEF_COLS + extra_cols)
    all_files_db = all_files_db.set_index(keys=["Campaign", "DUT", "Step"])

    # 2. load existing database file that is or is not empty
    loaded_db = None
    if os.path.exists(DataBaseFileLocation + DataBaseFile):
        print("Reading existing database file from", DataBaseFile)
        with pd.HDFStore(DataBaseFileLocation + DataBaseFile, mode="r+") as store:
            loaded_db = store["db"]

    # 3. either create database or extend existing database with the good images
    if loaded_db is None:
        loaded_db = all_files_db
    else:
        loaded_db = pd.concat([loaded_db, all_files_db], axis=0)
        loaded_db = loaded_db[~loaded_db.index.duplicated(keep="first")]

    # 4. get list of all bad pads which are not part of the database
    bad_files_db = all_files_db

    ## bad files that are in the new database but not in loaded
    bad_files_to_process = bad_files_db[~bad_files_db.index.isin(loaded_db.index)]
    bad_files_to_process = bad_files_to_process.assign(processed=False)
    return loaded_db, bad_files_to_process


