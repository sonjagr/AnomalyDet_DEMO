import pandas as pd
from helpers.plotting_helpers import *
from helpers.dataset_helpers import *
from helpers.annotation_helpers import closest_node, orig_box_to_coord, bbox_to_coord, intersection
np.random.seed(3)
import time

start_t = time.time()
def draw_patch(c, color, zorder):
    ax.add_patch(patches.Rectangle((c[0], c[1]), c[2]-c[0], c[3]-c[1],
                                   edgecolor=color, facecolor='none', zorder = zorder))

f = 'main_db_bb_crop.h5'
with pd.HDFStore(DataBaseFileLocation_local + f,  mode='r') as store:
        db = store.select('db_bb_crop')
        print(f'Reading {f}')

db_annotated = db
db_annotated['isctX'] = np.empty((len(db_annotated), 0)).tolist()
db_annotated['isctY'] = np.empty((len(db_annotated), 0)).tolist()
db_annotated['isct_dimX'] = np.empty((len(db_annotated), 0)).tolist()
db_annotated['isct_dimY'] = np.empty((len(db_annotated), 0)).tolist()
db_annotated['Anchor'] = np.empty((len(db_annotated), 0)).tolist()
db_annotated['Ai_dX'] = np.empty((len(db_annotated), 0)).tolist()
db_annotated['Ai_dY'] = np.empty((len(db_annotated), 0)).tolist()

draw = True
for image_index, row in db_annotated.iterrows():
    if db_annotated.at[image_index, "Normal"] == False:
        isctX, isctY, isct_dimX, isct_dimY, Anchor, Ai_dX, Ai_dY = [], [], [], [], [], [], []
        anchors = ANCHOR_GRID_STR.copy()
        anchors_num = ANCHOR_GRID.copy()
        if draw:
            fig, ax = plt.subplots(figsize=(16, 10))
            im = np.load(imgDir_local + '/' + image_index[0] + '/' + image_index[1] + '/' + row.FileName)
            im = cv2.cvtColor(im, cv2.COLOR_BAYER_RG2RGB)
            ax.imshow(im[0:PICTURESIZE_Y, 0:PICTURESIZE_X])
            ax.set_xticks(np.arange(80, PICTURESIZE_X, BOXSIZE_X), minor = True)
            ax.set_yticks(np.arange(80, PICTURESIZE_Y, BOXSIZE_Y), minor = True)
            ax.set_xticks(np.arange(0, PICTURESIZE_X, BOXSIZE_X))
            ax.set_yticks(np.arange(0, PICTURESIZE_Y, BOXSIZE_Y))
            ax.grid(which = 'minor', color='w', linewidth=0.3, linestyle ='-.', zorder = -2)
            ax.grid(which= 'major', color='w', linewidth=0.4, linestyle='--', zorder = -2)
        ins_list = []
        ins_whole_list = []
        for i in range(0, len(row.orig_boxX)):
            a = orig_box_to_coord(row.orig_boxX[i], row.orig_boxY[i])
            if draw:
                draw_patch(a, 'g', 2)
            for j in range(0, len(row.bound_boxX)):
                b = bbox_to_coord(row.bound_boxX[j], row.bound_boxY[j], row.bound_box_dimX[j], row.bound_box_dimY[j])
                if draw:
                    draw_patch(b, 'y', 1)
                ins = intersection(a,b)
                if not ins is None:
                    if abs(ins[2]-ins[0]) == BOXSIZE_X and abs(ins[3]-ins[1]) == BOXSIZE_Y:
                        ins_whole_list.append(ins)
                        if draw:
                            draw_patch(ins, 'r', 3)
                    elif abs(ins[2]-ins[0]) > SS_BOX_THRESH and abs(ins[3]-ins[1]) > SS_BOX_THRESH:
                        ins_list.append(ins)
                        if draw:
                            draw_patch(ins, 'r', 3)
        ins_whole_list = list(set(ins_whole_list))
        ins_list = list(set(ins_list))
        for ins in ins_whole_list:
            a = "%i-%i" % (ins[0] + int((ins[2]-ins[0])/2), ins[1] + int((ins[3]-ins[1])/2))
            isctX.append(ins[0])
            isctY.append(ins[1])
            isct_dimX.append(int((ins[2]-ins[0])))
            isct_dimY.append(int((ins[3]-ins[1])))
            distx, disty = 0, 0
            Ai_dX.append(distx)
            Ai_dY.append(disty)
            ix = anchors.index(a)
            Anchor.append(a)
            anchors.pop(ix)
            anchors_num.pop(ix)
        for ins in ins_list:
            a = [ins[0] + int((ins[2]-ins[0])/2), ins[1] + int((ins[3]-ins[1])/2)]
            isctX.append(ins[0])
            isctY.append(ins[1])
            isct_dimX.append(int((ins[2] - ins[0])))
            isct_dimY.append(int((ins[3] - ins[1])))
            x_ab, y_ab, distx, disty = closest_node(a, anchors_num)
            Ai_dX.append(distx)
            Ai_dY.append(disty)
            a = "%i-%i" % (x_ab, y_ab)
            Anchor.append(a)
            ix = anchors.index(a)
            anchors.pop(ix)
            anchors_num.pop(ix)

        db_annotated.at[image_index, "isctX"] = isctX
        db_annotated.at[image_index, "isctY"] = isctY
        db_annotated.at[image_index, "isct_dimX"] = isct_dimX
        db_annotated.at[image_index, "isct_dimY"] = isct_dimY
        db_annotated.at[image_index, "Anchor"] = Anchor
        db_annotated.at[image_index, "Ai_dX"] = Ai_dX
        db_annotated.at[image_index, "Ai_dY"] = Ai_dY
        plt.show()

#db_annotated.to_hdf(DataBaseFileLocation_gpu + 'db/three_annotations/main_db_bb_crop_ss.h5', key='db_bb_crop_ss', mode='w')
end_t = time.time()
print('Processing time was: ', end_t - start_t)