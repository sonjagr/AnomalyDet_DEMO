import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from scipy.ndimage.interpolation import rotate
import random
from old_codes.autoencoders import *
random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

ae = AutoEncoder()
print('Loading autoencoder...')
#ae.load('saved_class/model_AE_TQ3_500_to_500_epochs')
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

def encode(ae, imgs):
    img = imgs.reshape(-1, 2736, 3840,1)[:, :2720, :, :]
    t1 = time.time()
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = np.sqrt((decoded_img - img)**2)
    t2= time.time()
    ae_time = t2-t1
    return aed_img, ae_time

def split(img):

    t1 = time.time()
    split_img = tf.image.extract_patches(images=img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1], rates=[1, 1, 1, 1], padding='VALID')
    t2 = time.time()
    split_time = t2 - t1
    return split_img, split_time

savename = 'works2'
cont_epoch = 350
model = tf.keras.models.load_model('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_class/%s/cnn_%s_epoch_%i' % (savename, savename,cont_epoch))
#model = tf.keras.models.load_model('saved_class/cnn_%s_epoch_%i' % (savename,cont_epoch))


base_dir = 'db/'
dir_det = 'DET/'
MTN = 1

images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
#images_dir_loc = '/media/gsonja/Samsung_T5/ScratchDetection/MeasurementCampaigns/'
#images_dir_loc = 'F:/ScratchDetection/MeasurementCampaigns/'

images = 100

X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)[:images]
X_test_det_list = [images_dir_loc + s for s in X_test_det_list]
Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()[:images]

bce = tf.keras.losses.BinaryCrossentropy()

import time

loss = tf.keras.metrics.Mean()
encode_times = []
split_times = []
predict_times = []
tot_times = []

imgs = []
lbls = []
for i in range(0, images):
    y=Y_test_det_list[i].flatten()
    lbls = np.append(lbls, y, axis = 0)
    x = np.load(X_test_det_list[i])
    imgs = np.append(imgs, x)


encoded, ae_time = encode(ae, x)
split_img, split_time = split(encoded)
split_img = split_img.numpy().reshape(-1,160,160,1)
encode_times.append(ae_time)
split_times.append(split_time)
imgs.append(split_img)

imgs = np.array(imgs).reshape(-1, 160, 160, 1)
lbls = np.array(lbls)

t1 = time.time()
pred = model.predict(imgs)
t2 = time.time()

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(lbls, np.round(pred)).ravel()
print('TH  Test tn, fp, fn, tp:  ', tn, fp, fn, tp)

predict_times.append(t2 - t1)

print('Encode time ; ', np.sum(encode_times))
print('Split time ; ', np.sum(split_times))
print('Pred time ; ',t2 - t1 )


