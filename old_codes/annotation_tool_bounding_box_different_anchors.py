from tqdm import tqdm
import cv2
from helpers.annotation_helpers import *
import os.path
import matplotlib.patches as patches
from matplotlib import pyplot as plt

imgDir = imgDir_local
DataBaseFileLocation = DataBaseFileLocation_local

extra_cols = ["BoxX","BoxY","DimX","DimY"]

loaded_db, bad_files_to_process = get_bad_files_to_process(DataBaseFileLocation, DataBaseFile, imgDir, Campaigns, extra_cols)

def box_size(event):
    try:
        x_end.append(int(event.xdata))
        y_end.append(int(event.ydata))
    except:
        pass

def define_box(c1, c2):
    x1, y1 = c1
    x2,y2 = c2
    dx = abs(x1, x2)
    dy = abs(y1, y2)
    if y1 > y2 and x1 > x2:
        lr = c1
        ll = (x1-dx, y1)
        ul = (x1 - dx, y1 - dy)
        ur = (x1, y1 - dy)
    if y1 < y2 and x1 < x2:
        lr = c2
        ll = (x2-dx, y2)
        ul = (x2 - dx, y1 - dy)
        ur = (x1, y1 - dy)
    return ul, ur, ll, lr

def round_to_anchor(dX, dY):
    possible_cs = np.array([80, 160, 320, 480])
    print(dX, dY)
    dX, dY = int(dX), int(dY)
    ix = (np.abs(possible_cs - dX)).argmin()
    iy = (np.abs(possible_cs - dY)).argmin()
    return possible_cs[ix], possible_cs[iy]

def onclick_bb(event):
    try:
        x_click = int(event.xdata)
        y_click = int(event.ydata)
        click_ID = "%i-%i" % (x_click, y_click)
        x_prev = x_end[-1]
        y_prev = y_end[-1]
        if len(y_end) % 2 != 0:
            box_ID = "%i-%i" % (x_prev, y_prev)
            print('box ID', box_ID)
            update_canvas = False
            dX, dY = abs(x_prev-x_click), abs(y_prev-y_click)
            new_dX, new_dY = round_to_anchor(dX, dY)
            if event.dblclick:
                if overlap_click(click_ID, list(selected_boxes), new_dX, new_dY):
                    rem = match(box_ID, list(selected_boxes))[1]
                    selected_boxes[rem].remove()
                    del selected_boxes[rem]
                    update_canvas = True
            elif (len(list(selected_boxes)) == 0 or ((match(box_ID, list(selected_boxes))[0] is False))):
                dims[box_ID] = "%i-%i" % (new_dX, new_dY)
                selected_boxes[box_ID] = patches.Rectangle(
                    (x_click-abs(x_prev-x_click), y_click- abs(y_prev-y_click)), new_dX, new_dY, linewidth=0.6, edgecolor='r', facecolor='none')
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
    selected_boxes, dims = {}, {}
    x_end, y_end = [], []
    fig, ax = plt.subplots(figsize=(20, 12))
    cid = fig.canvas.mpl_connect('button_press_event', onclick_bb)
    bsz = fig.canvas.mpl_connect('button_press_event', box_size)
    im = np.load(filepath)
    im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
    ax.imshow(im[0:PICTURESIZE_Y, 0:PICTURESIZE_X])
    ax.set_xticks(np.arange(80, PICTURESIZE_X, 160), minor = True)
    ax.set_yticks(np.arange(80, PICTURESIZE_Y, 160), minor = True)
    ax.set_xticks(np.arange(0, PICTURESIZE_X, 160))
    ax.set_yticks(np.arange(0, PICTURESIZE_Y, 160))
    ax.grid(which = 'minor', color='w', linewidth=0.3, linestyle ='-.')
    ax.grid(which='major', color='w', linewidth=0.4, linestyle='--')
    #for point in ANCHOR_GRID:
    #    ax.plot(point[0], point[1], 'rx')
    plt.title(str(_img_index))
    plt.show()
    plt.pause(0.001)

    procd = True
    end = input(" Type 'x' to exit program, 's' to ignore previous annotation. Press enter to continue:  ")
    dims = {box: dims[box] for box in list(selected_boxes)}
    print(dims)
    boxesX, boxesY = [], []
    for box_ID in selected_boxes:
        b_x, b_y = tuple(box_ID.split("-"))
        boxesX.append(b_x)
        boxesY.append(b_y)

    dimsX, dimsY = [], []
    for box_ID in dims:
        d_x, d_y = tuple(dims.get(box_ID).split("-"))
        dimsX.append(d_x)
        dimsY.append(d_y)

    if "s" in end:
        print('Skip: previous image skipped')
        procd = False

    bad_files_to_process.at[_img_index, "BoxX"] = boxesX
    bad_files_to_process.at[_img_index, "BoxY"] = boxesY

    bad_files_to_process.at[_img_index, "DimX"] = dimsX
    bad_files_to_process.at[_img_index, "DimY"] = dimsY

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