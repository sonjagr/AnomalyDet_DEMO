import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from common import *
from helpers.annotation_helpers import get_bad_files_to_process
import os.path
import matplotlib.patches as patches
from matplotlib import pyplot as plt

imgDir = imgDir_local
DataBaseFileLocation = DataBaseFileLocation_local

extra_cols = ["BoxX","BoxY"]

loaded_db, bad_files_to_process = get_bad_files_to_process(DataBaseFileLocation, DataBaseFile, imgDir, Campaigns, extra_cols)

#5. loop over all files to process and perform the selection
def onclick_original(event):
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

print("Number of unprocessed images: ", len(process_indexes))
for _c, _img_index in enumerate(tqdm(process_indexes)):
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

    procd = True
    end = input(" Type 'x' to exit program, 's' to ignore previous annotation. Press enter to continue:  ")

    boxesX, boxesY = [], []
    for box_ID in selected_boxes:
        b_x, b_y = tuple(box_ID.split("-"))
        boxesX.append(b_x)
        boxesY.append(b_y)
    if "s" in end:
        print('Skip: previous image skipped')
        procd = False

    bad_files_to_process.at[_img_index, "BoxX"] = boxesX
    bad_files_to_process.at[_img_index, "BoxY"] = boxesY
    bad_files_to_process.at[_img_index, "processed"] = procd
    if "x" in end:
        print('Quit: annotations will be saved')
        break

#6. Determine which files were processed and can be joined with the database
bad_files_processed = bad_files_to_process[bad_files_to_process.processed==True]
bad_files_processed = bad_files_processed.drop(columns=["processed"])
print(bad_files_processed)

loaded_db = pd.concat([loaded_db, bad_files_processed], axis = 0)

#7.: write out database
with pd.HDFStore(DataBaseFileLocation + DataBaseFile, mode="w") as store:
    store["db"] = loaded_db