from tqdm import tqdm
import cv2
from helpers.annotation_helpers import *
import os.path
import matplotlib.patches as patches
from matplotlib import pyplot as plt

imgDir = imgDir_local
DataBaseFileLocation = DataBaseFileLocation_local

extra_cols = ["BoxX","BoxY","ClickX","ClickY"]

loaded_db, bad_files_to_process = get_bad_files_to_process(DataBaseFileLocation, DataBaseFile, imgDir, Campaigns, extra_cols)

#5. loop over all files to process and perform the selection
def onclick_ol(event):
    try:
        x_click = event.xdata
        y_click  = event.ydata
        x_top, y_top, _, _ = closest_node([x_click, y_click], ANCHOR_GRID_OFFSET_X_Y)
        print('a grid', len(np.unique(ANCHOR_GRID_OFFSET_X_Y, axis=0)))
        x_top = x_top - 80
        y_top = y_top - 80
        box_ID = "%i-%i" % (x_top, y_top)
        box_ID_1 = "%i-%i" % (x_top-80, y_top-80)
        box_ID_2 ="%i-%i" % (x_top-80, y_top+80)
        box_ID_3 = "%i-%i" % (x_top+80, y_top+80)
        box_ID_4 = "%i-%i" % (x_top+80, y_top-80)
        box_ID_5 = "%i-%i" % (x_top+80, y_top)
        box_ID_6 = "%i-%i" % (x_top, y_top-80)
        box_ID_7 = "%i-%i" % (x_top-80, y_top)
        box_ID_8 = "%i-%i" % (x_top, y_top+80)
        click_ID = "%i-%i" % (x_click, y_click)
        clicks[box_ID]  =click_ID
        update_canvas = False
        if event.dblclick:
            if box_ID in selected_boxes:
                #print('boef ', clicks)
                selected_boxes[box_ID].remove()
                del selected_boxes[box_ID]
                update_canvas = True
        else:
            forb = [box_ID, box_ID_1, box_ID_2, box_ID_3, box_ID_4, box_ID_5, box_ID_6, box_ID_7, box_ID_8]
            t = [i for i in list(selected_boxes) if i in forb]
            if len(list(selected_boxes)) == 0 or (len(t)==0):
                selected_boxes[box_ID] = patches.Rectangle(
                    (x_top, y_top), BOXSIZE_X, BOXSIZE_Y, linewidth=1, edgecolor='r', facecolor='none')
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
    selected_boxes, clicks = {}, {}
    fig, ax = plt.subplots(figsize=(16, 10))
    cid = fig.canvas.mpl_connect('button_press_event', onclick_ol)
    im = np.load(filepath)
    im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
    ax.imshow(im[0:PICTURESIZE_Y, 0:PICTURESIZE_X])
#    for point in np.unique(ANCHOR_GRID_OFFSET_X_Y, axis=0):
#        ax.plot(point[0], point[1], 'rx')
    plt.title(str(_img_index))
    plt.show()
    plt.pause(0.001)

    procd = True
    end = input(" Type 'x' to exit program, 's' to ignore previous annotation. Press enter to continue:  ")
    boxes = list(selected_boxes)
    clicks = {box: clicks[box] for box in boxes}
    boxesX, boxesY = [], []
    for box_ID in selected_boxes:
        b_x, b_y = tuple(box_ID.split("-"))
        boxesX.append(b_x)
        boxesY.append(b_y)

    clicksX, clicksY = [], []
    for box_ID in clicks:
        c_x, c_y = tuple(clicks.get(box_ID).split("-"))
        clicksX.append(c_x)
        clicksY.append(c_y)
    if "s" in end:
        print('Skip: previous image skipped')
        procd = False

    bad_files_to_process.at[_img_index, "BoxX"] = boxesX
    bad_files_to_process.at[_img_index, "BoxY"] = boxesY

    bad_files_to_process.at[_img_index, "ClickX"] = clicksX
    bad_files_to_process.at[_img_index, "ClickY"] = clicksY

    bad_files_to_process.at[_img_index, "processed"] = procd
    if "x" in end:
        print('Quit: annotations will be saved')
        break

#6. Determine which files were processed and can be joined with the database
bad_files_processed = bad_files_to_process[bad_files_to_process.processed==True]
bad_files_processed = bad_files_processed.drop(columns=["processed"])

loaded_db = pd.concat([loaded_db, bad_files_processed], axis = 0)

#7.: write out database
with pd.HDFStore(DataBaseFileLocation + DataBaseFile, mode="w") as store:
    store["db"] = loaded_db