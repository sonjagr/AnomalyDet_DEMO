import numpy as np
import os
import cv2

from common import *
from matplotlib import pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import ImageGrid

def plot_annotated_image(file, boxX, boxY, dimX, dimY,  saveloc):
    image = np.load(file)
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
    #plt.savefig(saveloc, dpi=600)
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
    #plt.savefig(saveloc, dpi=600)
    plt.show()

def grid_plotter(files, saveloc):
    fig = plt.figure(figsize=(5.*1.4, 5.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(10, 5),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, files):
        image = np.load(os.path.join(imgDir_pc,im))
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
        # Iterating over the grid returns the Axes.
        ax.imshow(image)
        ax.axis('off')
    plt.tight_layout()
    #plt.savefig(saveloc, dpi=600)
    plt.show()
