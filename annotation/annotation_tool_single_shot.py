from tqdm import tqdm
import cv2
from helpers.annotation_helpers import *
import os.path
import matplotlib.patches as patches
from matplotlib import pyplot as plt

imgDir = imgDir_local
DataBaseFileLocation = DataBaseFileLocation_local

extra_cols = ["ClickX","ClickY", "BoxX", "BoxY", "dX","dY"]

loaded_db, bad_files_to_process = get_bad_files_to_process(DataBaseFileLocation, DataBaseFile, imgDir, Campaigns, extra_cols)

#5. loop over all files to process and perform the selection

def onclick_ss(event):
    try:
        x_click = int(event.xdata)
        y_click = int(event.ydata)
        x_ab, y_ab, _, _ = closest_node([x_click, y_click], ANCHOR_GRID)
        x_top = x_click-(BOXSIZE_X/2)
        y_top = y_click-(BOXSIZE_Y/2)
        box_ID = "%i-%i" % (x_ab-(BOXSIZE_X/2), y_ab-(BOXSIZE_X/2))
        print(box_ID)
        click_ID = "%i-%i" % (x_click, y_click)
        print('box_ID: ', box_ID)
        update_canvas = False
        if event.dblclick:
            if overlap_click(click_ID, list(selected_boxes)):
                rem = match(box_ID, list(selected_boxes))[1]
                selected_boxes[rem].remove()
                del selected_boxes[rem]
                update_canvas = True
        else:
            # boxes inside figure
            if x_click >= 80 and x_click <= PICTURESIZE_X - 80 and y_click >= 80 and y_click <= PICTURESIZE_Y - 80 and overlap_click(click_ID, list(selected_boxes)) == False:
                # only one box per anchor point
                if not box_ID in selected_boxes:
                    clicks[box_ID] = click_ID
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
    dxs, dys = [], []
    fig, ax = plt.subplots(figsize=(20, 12))
    cid = fig.canvas.mpl_connect('button_press_event', onclick_ss)
    im = np.load(filepath)
    im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
    ax.imshow(im[0:PICTURESIZE_Y, 0:PICTURESIZE_X])
    for point in ANCHOR_GRID:
        ax.plot(point[0], point[1], 'rx')
    plt.title(str(_img_index))
    plt.show()
    plt.pause(0.001)

    procd = True
    end = input(" Type 'x' to exit program, 's' to ignore previous annotation. Press enter to continue:  ")

    clicks = {box: clicks[box] for box in list(selected_boxes)}
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

    _, _, dxs, dys = clicks_to_boxes(clicksX, clicksY, anchors=ANCHOR_GRID)

    bad_files_to_process.at[_img_index, "ClickX"] = clicksX
    bad_files_to_process.at[_img_index, "ClickY"] = clicksY
    bad_files_to_process.at[_img_index, "BoxX"] = boxesX
    bad_files_to_process.at[_img_index, "BoxY"] = boxesY
    bad_files_to_process.at[_img_index, "dX"] = dxs
    bad_files_to_process.at[_img_index, "dY"] = dys
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