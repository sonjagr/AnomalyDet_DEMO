import matplotlib.pyplot as plt
import numpy as np

def read_npy_file(item):
    data = np.load(item)
    data = np.expand_dims(data, axis=2)
    return data.astype(np.float32)

path_to_image = 'C:/Users/sgroenro/cernbox/WINDOWS/Desktop/examples/raw_example/rescan_pad44_step42.npy'
image = read_npy_file(path_to_image)

plt.imshow(image)
plt.show()

import cv2
image_rgb = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BAYER_RG2RGB)
plt.imshow(image_rgb)
plt.show()
