import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

bayer = np.load('/media/gsonja/Samsung_T5/ScratchDetection/MeasurementCampaigns/EndOf2021_PM8/8inch_444ch_N4788_2/initial_scan_pad360_step351.npy')

plt.imshow(bayer)
plt.show()

rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2RGB)

plt.imshow(rgb)
plt.show()

def rgb2bayer(rgb):
    (height, width) = rgb.shape[:2]
    (R, G, B) = cv2.split(rgb)

    bayer = np.empty((height, width), np.uint8)

    # strided slicing for this pattern:
    #   R G
    #   B R
    bayer[0::2, 0::2] = R[0::2, 0::2]  # top left
    bayer[0::2, 1::2] = G[0::2, 1::2]  # top right
    bayer[1::2, 0::2] = G[1::2, 0::2]  # bottom left
    bayer[1::2, 1::2] = B[1::2, 1::2]  # bottom right
    return bayer

bayer2 = rgb2bayer(rgb)
plt.imshow(bayer2)
plt.show()