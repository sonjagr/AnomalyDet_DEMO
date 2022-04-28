import numpy as np
from common import *
from matplotlib import pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2

def plot_annotated_image(file, boxX, boxY, dimX, dimY,  saveloc):
    image = np.load(file)
    #image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
    fig, ax = plt.subplots()
    ax.imshow(image[0:PICTURESIZE_Y, 0:PICTURESIZE_X])
    for i in range(0,len(boxX)):
        rec = patches.Rectangle((int(boxX[i]), int(boxY[i])), int(dimX), int(dimY), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rec)
    #plt.savefig(saveloc, dpi=600)
    plt.show()

def plot_image(file, boxx, boxy, saveloc):
    image = np.load(file)
    image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
    fig, ax = plt.subplots()
    ax.imshow(image[0:PICTURESIZE_Y, 0:PICTURESIZE_X])
    plt.savefig(saveloc, dpi=600)
    plt.show()

def plot_grid_image(file, saveloc):
    image = np.load(file)
    image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
    for y in range(0, 17):
        cv2.line(image, (0, y * BOXSIZE_Y), (PICTURESIZE_X, y * BOXSIZE_Y), (255, 0, 0), 10)
    for x in range(0, 24):
        cv2.line(image, (x * BOXSIZE_X, 0), (x * BOXSIZE_X, PICTURESIZE_Y), (255, 0, 0), 10)
    fig, ax = plt.subplots()
    ax.imshow(image[0:PICTURESIZE_Y, 0:PICTURESIZE_X])
    plt.savefig(saveloc, dpi=600)
    plt.show()

def grid_plotter(files, saveloc):
    fig = plt.figure(figsize=(5.*1.4, 5.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(10, 5),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, files):
        Camp, dut = "September2021_PM8", "8inch_198ch_N3311_6"
        image = np.load(imgDir_local+'/'+im)
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
        # Iterating over the grid returns the Axes.
        ax.imshow(image)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(saveloc, dpi=600)
    plt.show()

boxx = ['1120', '960', '960', '960', '960', '960', '960', '960']
boxy = ['1920', '1920', '1760', '1600', '1440', '1280', '1120', '960']
name = imgDir_local+"September2021_PM8/"+"8inch_198ch_N3311_6/"+"initial_scan_pad57_step167.npy"

#plot_annotated_image(name, boxx, boxy, 'C:/Users/sgroenro/images/'+'example_ann_grid_3.png')