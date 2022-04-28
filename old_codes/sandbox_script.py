from helpers.plotting_helpers import *
from helpers.dataset_helpers import *

'''
def coords_to_boxid(x,y):
    boxids = []
    for i in range(len(x)):
        x_box = x[i]/BOXSIZE_X
        y_box = y[i]/BOXSIZE_Y
        ys = 17
        y_part = y_box % ys
        boxids.append(x_box + y_part)
    return boxids

with pd.HDFStore(DataBaseFileLocation_local + 'Winter2022_ALPS_database_new.h5', mode='r') as store:
    db_to = store.select("db")

db = db_to.copy()

norm_db = db[db.Normal == True][:10]
def_db = db[db.Normal == False][:10]

db = pd.concat([norm_db, def_db])
db = db.drop(["Normal"], axis =1)
db = db.rename(columns={"BoxX": "BoxX_hat", "BoxY": "BoxY_hat"})
db["Validated"], db["BoxX"], db["BoxY"], db["Processed"] = np.nan, np.nan, np.nan, False

db['Normal_hat'] = True
db.loc[db['BoxX_hat'].str.len() > 0, 'Normal_hat'] = False

print(db.columns)
print(db[['Normal_hat', 'BoxX_hat']])

with pd.HDFStore(DataBaseFileLocation_local + 'predictions.h5', mode="w") as store:
    store["db_hat"] = db

#axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
#bnext = Button(axnext, 'Next')
#axquit = plt.axes([0.21, 0.05, 0.1, 0.075])
#bquit = Button(axquit, 'Quit')
#bquit.on_clicked(on_quit_button_clicked)

'''

'''
def plot_annotations(df_obj):
    boxy = df_obj.BoxY.values[0]
    boxx = df_obj.BoxX.values[0]
    name = df_obj.FileName.values
    camp = df_obj.index.values[0][0]
    dut = df_obj.index.values[0][1]
    path = imgDir_local + str(camp) + '/' + str(dut) +'/' + name
    path = str(path[0])
    image = np.load(path)
    #image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
    fig, ax = plt.subplots()
    ax.imshow(image[0:PICTURESIZE_Y, 0:PICTURESIZE_X])
    for i in range(0, len(boxx)):
        rec = patches.Rectangle((int(boxx[i]), int(boxy[i])), BOXSIZE_X, BOXSIZE_Y, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rec)
    #plt.savefig(saveloc, dpi=600)
    plt.show()

imgDir = imgDir_local
DataBaseFileLocation = 'C:/Users/sgroenro/Downloads/'
DataBaseFile = 'November2021(1).h5'
loaded_db = None
if os.path.exists(DataBaseFileLocation + DataBaseFile):
    print("Reading existing database file from", DataBaseFile)
    with pd.HDFStore(DataBaseFileLocation + DataBaseFile, mode="r+") as store:
        loaded_db = store["db"]
print(loaded_db[len(loaded_db.BoxY.values.tolist()) > 3])
#idx = pd.IndexSlice
#t = loaded_db.loc[idx[:, :, 33], :]
#plot_annotations(t)
#print(t)

s_x, s_y = 315, 1232 #upper corner
c_x, c_y =  419, 1343 #click

if (s_x <= c_x <= (s_x+160)) and (s_y <= c_y <= (s_y + 160)):
    print('overlapppp')
print('noo')
'''

'''
ANCHOR_GRID = []

for row in range(int(BOXSIZE_Y/2), PICTURESIZE_Y+int(BOXSIZE_Y/2), BOXSIZE_Y):
    for col in range(int(BOXSIZE_X/2), PICTURESIZE_X+int(BOXSIZE_Y/2), BOXSIZE_X):
        ANCHOR_GRID.append([col, row])
print(17*24)
print(ANCHOR_GRID)
'''
'''
def pointInRect(point,rect):
    x1, y1= rect
    w,h = 160, 160
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False

def match(box_id, boxes):
    c_x, c_y = tuple(box_id.split("-"))
    c_x, c_y = int(c_x), int(c_y)
    for k in range(0,len(boxes)):
        b_x, b_y =  tuple(boxes[k].split("-"))
        b_x, b_y = int(b_x), int(b_y)
        if pointInRect((c_x, c_y), (b_x,b_y)):
            return True
    return False

def match2(box_id, boxes):
    expanded_box_id = []
    c_x, c_y = tuple(box_id.split("-"))
    c_x, c_y = int(c_x), int(c_y)
    for k in range(0,4):
        c_x, c_y_n = c_x, c_y
        if k == 0:
            c_x = c_x + 0
        else:
            c_x = c_x + 1
        for j in range(0,4):
            if j == 0:
                c_y_n = c_y_n + 0
            else:
                c_y_n = c_y_n + (1)
            expanded_box_id.append("%i-%i" % (c_x, c_y_n))
    print('exp box ' , expanded_box_id)
    print('boxes ', boxes)
    t = [i for i in boxes if i in expanded_box_id]
    if  len(t) == 0:
        return False
    else:
        return True

box_ID ='162-330'
boxes =['1653-678', '166-160', '161-160']

t = match(box_ID, boxes)
print(t)


def overlap(box_id, boxes):
    c_x, c_y = tuple(box_id.split("-"))
    c_x, c_y = int(c_x), int(c_y)
    for k in range(0,len(boxes)):
        b_x, b_y =  tuple(boxes[k].split("-"))
        b_x, b_y = int(b_x), int(b_y)
        x1min, y1min = c_x, c_y
        x1max, y1max= c_x+160, c_y+160
        x2min, y2min = b_x, b_y
        x2max, y2max= b_x+160, b_y+160
        if (x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max):
            return True
    return False

box_ID ='162-330'
boxes =['1653-678', '166-160', '161-160']

t = overlap(box_ID, boxes)
print(t)


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

'''
bbsX, bbsY, dimsX, dimsY = ['832'], ['160'], ['351'], ['241']

def doOverlap(l1, r1, l2, r2):
    if (l1[0] == r1[0] or l1[1] == r1[1] or l2[0] == r2[0] or l2[1] == r2[1]):
        return False
    if (l1[0] >= r2[0] or l2[0] >= r1[0]):
        return False
    if (r1[1] >= l2[1] or r2[1] >= l1[1]):
        return False
    return True

def overlapp(a, b):
    if (a[1] >= b[2] or a[0] >= b[3] or a[2] <= b[1] or a[3] <= b[0]):
        return False
    return True

def bb_to_sb(bbsX, bbsY, dimsX, dimsY):
    anchors = ANCHOR_GRID
    selected_boxes = {}
    for i in range(0,len(bbsX)):
        rr1 = [int(bbsX[i]), int(bbsY[i]),int(bbsX[i])+int(dimsX[i]), int(bbsY[i])+int(dimsY[i])]
        for j in range(0,int(len(anchors)/2)):
            rr2 = [anchors[j][0]-80, anchors[j][1]-80, anchors[j][0] + 80, anchors[j][1] + 80]
            overlap = overlapp(rr1, rr2)
            if overlap == True:
                box_ID  = "%i-%i" % (anchors[j][0]-80, anchors[j][1]-80)
                selected_boxes[box_ID] = patches.Rectangle(
                    (anchors[j][0]-80, anchors[j][1]-80), BOXSIZE_X, BOXSIZE_Y, linewidth=1, edgecolor='r', facecolor='none')
    return selected_boxes

ss = bb_to_sb([60], [60], [200], [500])
print(ss)