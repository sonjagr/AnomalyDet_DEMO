import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from common import *
from scipy.ndimage.interpolation import rotate
import random, argparse
from autoencoders import *
random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

ae = AutoEncoder()
print('Loading autoencoder...')
ae.load('saved_class/model_AE_TQ3_500_to_500_epochs')
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')

@tf.function
def compute_loss_test(model, x, y_ref):
    y = model(x, training = False)
    reconstruction_error = bce(y_ref, y)
    return reconstruction_error

def rotate_img(img):
    img = img.reshape(160,160)
    rot_angle = random.choice([90, 180, 270])
    rot = rotate(img, rot_angle)
    return rot.flatten()

def encode_split(ae, img):
    img = img.numpy().reshape(1, 2736, 3840,1)[:, :2720, :, :]
    t1 = time.time()
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = np.sqrt((decoded_img - img)**2)
    t2= time.time()
    ae_time = t2-t1
    t1 = time.time()
    split_img = tf.image.extract_patches(images=aed_img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1],rates=[1, 1, 1, 1], padding='VALID')
    t2 = time.time()
    split_time = t2-t1
    return split_img.numpy().reshape(17 * 24, 160 * 160), ae_time, split_time

#savename = 'works3_bs128'
savename = 'works2'
cont_epoch = 350
model = tf.keras.models.load_model('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_class/%s/cnn_%s_epoch_%i' % (savename, savename,cont_epoch))
#model = tf.keras.models.load_model('saved_class/cnn_%s_epoch_%i' % (savename,cont_epoch))


base_dir = 'db/'
dir_det = 'DET/'
MTN = 1

images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
#images_dir_loc = '/media/gsonja/Samsung_T5/ScratchDetection/MeasurementCampaigns/'

X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)[:10]
X_test_det_list = [images_dir_loc + s for s in X_test_det_list]

Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()[:10]

bce = tf.keras.losses.BinaryCrossentropy()

print(len(X_test_det_list), len(Y_test_det_list))

import time
import timeit
loss = tf.keras.metrics.Mean()
encode_times = []
split_times = []
predict_times = []
tot_times = []
for i in range(0,10):
    #print(i)
    for i in range(0,10):
        y=Y_test_det_list[i]
        x = np.load(X_test_det_list[i])
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        tot1 = time.time()
        img, ae_time, split_time = encode_split(ae, x)
        img = img.reshape(-1,160,160,1)
        encode_times.append(ae_time)
        split_times.append(split_time)
        t1 = time.time()
        pred = model.predict(img)
        tot2 = time.time()
        t2 = time.time()
        tot_times.append(tot2-tot1)
        predict_times.append(t2-t1)

print('Encode time ; ', np.mean(encode_times))
print('Split time ; ', np.mean(split_times))
print('Pred time ; ', np.mean(predict_times))
print('tot time ; ', np.mean(tot_times))


