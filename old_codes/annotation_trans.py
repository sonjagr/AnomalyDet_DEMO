from helpers import plotting_helpers
from common import *
import pandas as pd

with pd.HDFStore("C:/Users/sgroenro/codes/anomaly-detection/db/main/testing_bb",  mode='r') as store:
    db = store.select("db")
db_def = db[db.Normal == False]
print(db_def)

example = db_def.loc[("Fall2021_PM8", "8inch_198ch_N4790_1", 30)]
print(example)

box_ul = (int(example.BoxX[0]), int(example.BoxY[0]))
box_dim = (int(example.DimX[0]), int(example.DimY[0]))
print(box_ul)

file = imgDir_local+"Fall2021_PM8/"+"8inch_198ch_N4790_1/"+example.FileName
plotting_helpers.plot_annotated_image(file, example.BoxX, example.BoxY, example.DimX, example.DimY, None)