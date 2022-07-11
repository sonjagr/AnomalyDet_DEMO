import numpy as np
import random
from AE.autoencoders2 import *
from sklearn.metrics import confusion_matrix
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ae = AutoEncoder()
import matplotlib.pyplot as plt
import time
random.seed(42)

ae = AutoEncoder()
savename = 'works2'
cont_epoch = 168
base_dir = '../db/'
dir_det = 'DET/'

computer = 'gpu'

if computer == 'gpu':
    ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')
    model = tf.keras.models.load_model(
        '/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/testing_brightness/cnn_testing_brightness_epoch_168')
    images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
if computer == 'local':
    ae.load('saved_class/model_AE_TQ3_500_to_500_epochs')
    model = tf.keras.models.load_model('saved_class/cnn_%s_epoch_%i' % (savename, cont_epoch))
    images_dir_loc = '/media/gsonja/Samsung_T5/ScratchDetection/MeasurementCampaigns/'
    # images_dir_loc = 'F:/ScratchDetection/MeasurementCampaigns/'

def encode(ae, imgs):
    img = imgs.reshape(-1, 2736, 3840,1)[:, :2720, :, :]
    t1 = time.time()
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    #print('decoded shape: ', decoded_img.shape)
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

images = 32

X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)[:images]
X_test_det_list = [images_dir_loc + s for s in X_test_det_list]
Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()[:images]

encode_times = []
split_times = []
predict_times = []
tot_times = []

imgs = []
lbls = []
for i in range(0, images):
    print(i)
    y=Y_test_det_list[i].flatten()
    lbls = np.append(lbls, y, axis = 0)
    x = np.load(X_test_det_list[i])
    imgs = np.append(imgs, x)
print('Images red')

encoded, ae_time = encode(ae, imgs)
print('Encoded')
split_img, split_time = split(encoded)
print('Splitted size', split_img.shape)
split_img = split_img.numpy().reshape(-1,160,160,1)

defs = 0
for i in range(0, 8160):
    if lbls[i] == 1:
        defs = defs+1
        plt.imshow(split_img[i].reshape(160,160))
        plt.show()
print(defs)
encode_times.append(ae_time)
split_times.append(split_time)

imgs = np.array(split_img).reshape(-1, 160, 160, 1)
print(imgs.shape)
lbls = np.array(lbls)

t1 = time.time()
pred = model.predict(imgs)
t2 = time.time()

tn, fp, fn, tp = confusion_matrix(lbls, np.round(pred)).ravel()
print('TH  Test tn, fp, fn, tp:  ', tn, fp, fn, tp)

predict_times.append(t2 - t1)

print('Encode time ; ', np.sum(encode_times))
print('Split time ; ', np.sum(split_times))
print('Pred time ; ',t2 - t1 )


