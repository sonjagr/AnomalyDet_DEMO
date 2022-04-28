import os, sys, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import sklearn
from sklearn import metrics
from common import *
import os.path, sys
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

## reading database containing the predictions
predictions = DataBaseFileLocation_local + 'predictions.h5'

if not os.path.exists(predictions):
    sys.exit(predictions+" as file where predictions are stored does not exists.")
else:
    with pd.HDFStore(predictions,  mode='r') as store:
        db = store.select("db_hat")

proc = len(db[db.Processed == True])
if proc > 0:
    processed = input(" Type 're' to re-validate, anything else to continue validation: ")
    if processed == 're':
        index = input("Give index(es) to re-annotate or type 'all': ")
        if index == 'all':
            # reset validation annotations
            db['Processed'], db['Validated'], db['BoxX'], db['BoxY'], db['Boxids'], db['Boxids_pred'] = False, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            for reset_i in index:
                db.at[reset_i,'Processed'], db.at[reset_i, 'Validated'],  db.at[reset_i, 'BoxX'], db.at[reset_i, 'BoxY'], db.at[reset_i, 'Boxids'], db.at[reset_i, 'Boxids_pred'] = False, np.nan, np.nan, np.nan, np.nan, np.nan

pred_norm_valid = db[(db.Normal_hat == True)].sample(frac=0.2, random_state = 1)
pred_norm_valid = pred_norm_valid[pred_norm_valid.Processed != True].index

pred_norm = db[db.Normal_hat == True].index
pred_def = db[(db.Normal_hat == False) & (db.Processed != True)].index

validation_indexes = pred_def.append(pred_norm_valid)

def coords_to_boxid(x,y):
    boxids = []
    for i in range(len(x)):
        y_box = int(x[i])/BOXSIZE_X
        x_box = int(y[i])/BOXSIZE_Y
        if y_box <= YS:
            boxids.append(int((XS * x_box) + (y_box)))
        else:
            boxids.append(int((XS*x_box) + (y_box % YS)))
    return boxids

def onclick(event):
    try:
        x_bottom = int(event.xdata/BOXSIZE_X)*BOXSIZE_X
        y_bottom = int(event.ydata/BOXSIZE_Y)*BOXSIZE_Y
        box_ID = "%i-%i" % (x_bottom, y_bottom)
        if str(box_ID) in predicted_boxes:
            color = 'lawngreen'
        if str(box_ID) not in predicted_boxes:
            color = 'r'
        update_canvas = False
        if event.dblclick:
            if box_ID in selected_boxes:
                selected_boxes[box_ID].remove()
                del selected_boxes[box_ID]
                update_canvas = True
        else:
            if not box_ID in selected_boxes:
                selected_boxes[box_ID] = patches.Rectangle(
                    (x_bottom, y_bottom), BOXSIZE_X, BOXSIZE_Y, linewidth=1, edgecolor=color, facecolor='none')
                ax.add_patch(selected_boxes[box_ID])
                update_canvas = True
        if update_canvas:
            plt.draw()
            plt.pause(0.001)
    except:
        pass

db['BoxX'] = db['BoxX'].astype(object)
db['BoxY'] = db['BoxY'].astype(object)

db['Boxids'] = db['Boxids'].astype(object)
db['Boxids_pred'] = db['Boxids_pred'].astype(object)

print('Validate %s anomalous images' % len(validation_indexes))
for _c, _img_index in enumerate(tqdm(validation_indexes)):
    if _c > 0:
        end = input(" Type 'x' to exit program, enter to continue: ")
        if end == "x":
            break
    filepath = os.path.join(imgDir_local, _img_index[0], _img_index[1], db.loc[_img_index, "FileName"])
    selected_boxes = {}
    BoxX_pred = db.at[_img_index, "BoxX_hat"]
    BoxY_pred = db.at[_img_index, "BoxY_hat"]
    predicted_boxes = []
    fig, ax = plt.subplots(figsize=(20, 12))
    db.at[_img_index, "Boxids_pred"] = coords_to_boxid(BoxX_pred, BoxY_pred)
    for i in range(0,len(BoxY_pred)):
        predicted_boxes.append("%i-%i" % (int(BoxX_pred[i]), int(BoxY_pred[i])))
        rec = patches.Rectangle((int(BoxX_pred[i]), int(BoxY_pred[i])), BOXSIZE_X, BOXSIZE_Y, linewidth=1, edgecolor='y', facecolor='none')
        ax.add_patch(rec)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    im = np.load(filepath)
    im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
    ax.imshow(im[0:PICTURESIZE_Y, 0:PICTURESIZE_X])
    plt.title('Validate anomalies', fontsize = 14)
    if db.at[_img_index, "Normal_hat"] == False:
        legend_elements = [patches.Patch(facecolor='None', edgecolor='r',  label='FN'), patches.Patch(facecolor='None', edgecolor='lawngreen', label='TP'), patches.Patch(facecolor='None', edgecolor='yellow', label='P')]
    if db.at[_img_index, "Normal_hat"] == True:
        legend_elements = [patches.Patch(facecolor='None', edgecolor='r', label='FN')]
    plt.legend(handles=legend_elements)
    plt.show()
    plt.pause(0.001)
    boxesX, boxesY = [], []
    for box_ID in selected_boxes:
        b_x, b_y = tuple(box_ID.split("-"))
        boxesX.append(b_x)
        boxesY.append(b_y)
    db.at[_img_index, "Boxids"] = coords_to_boxid(boxesX, boxesY)
    db.at[_img_index, "BoxX"] = boxesX
    db.at[_img_index, "BoxY"] = boxesY
    #if len(boxesX) > 0:
    #    db.at[_img_index, "Normal"] = False
    #else:
    #    db.at[_img_index, "Normal"] = True
    ## compare human and machine annotations
    boxesY_hat = db.at[_img_index, "BoxY_hat"]
    boxesX_hat = db.at[_img_index, "BoxX_hat"]
    boxesY_hat, boxesX_hat = [float(i) for i in boxesY_hat], [float(i) for i in boxesX_hat ]
    boxesY, boxesX = [float(i) for i in boxesY], [float(i) for i in boxesX]
    if sorted(boxesY_hat) != sorted(boxesY) or sorted(boxesX_hat) != sorted(boxesX):
        print('Not validated: difference detected')
        db.at[_img_index, "Validated"] = 1
    if sorted(boxesY_hat) == sorted(boxesY) and sorted(boxesX_hat) == sorted(boxesX):
        print('Validated')
        db.at[_img_index, "Validated"] = 2
    db.at[_img_index, "Processed"] = True

processed_db = db[db.Processed == True]

def evaluate(true, preds, norm):
    pred_imgs = []
    true_imgs = []
    for i in range(0,len(preds)):
        true_m = np.zeros((YS*XS))
        pred_m = np.zeros((YS*XS))
        true_m[true[i]] = 1
        pred_m[preds[i]] = 1
        pred_imgs = np.append(pred_imgs, pred_m)
        true_imgs = np.append(true_imgs, true_m)
    cm = sklearn.metrics.confusion_matrix(pred_imgs, true_imgs, normalize = norm)
    return cm

preds = processed_db.Boxids_pred.values.tolist()
true = processed_db.Boxids.values.tolist()

tn, fp, fn, tp = evaluate(true, preds, None).ravel()
print( tn, fp, fn, tp )
sens, spec = tp/(tp + fn), tn/(tn + fp)
print(sens, spec)

with pd.HDFStore(predictions, mode="w") as store:
    store["db_hat"] = db
