import matplotlib.pyplot as plt
import numpy as np

def read_npy_file(item):
    data = np.load(item)
    data = np.expand_dims(data, axis=2)
    return data

path_to_image = 'C:/Users/sgroenro/cernbox/WINDOWS/Desktop/examples/raw_example/rescan_pad44_step42.npy'
image = read_npy_file(path_to_image)

plt.imshow(image)
plt.show()

import cv2
test1 = cv2.split(cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB), cv2.COLOR_RGB2HLS_FULL))[0]
test2 = cv2.split(cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB), cv2.COLOR_RGB2HLS_FULL))[1]
test3 = cv2.split(cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB), cv2.COLOR_RGB2HLS_FULL))[2]
test4 = cv2.split( cv2.cvtColor( cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB), cv2.COLOR_RGB2YUV))[0]
test5 = cv2.split(cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB), cv2.COLOR_RGB2YCrCb))[2]

testlist = [test1, test2, test3, test4, test5]
for i in testlist:
    plt.imshow(i)
    plt.show()



