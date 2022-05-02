import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from helpers.dataset_helpers import create_cnn_dataset
from autoencoders2 import *
from common import *
from scipy.ndimage.interpolation import rotate
import random, argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn import metrics
import tensorflow as tf
random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

tf.keras.backend.clear_session()

ae = AutoEncoder()
print('Loading autoencoder...')
ae.load('saved_class/model_AE_TQ3_500_to_500_epochs')
#ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')

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
#model = tf.keras.models.load_model('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_class/%s/cnn_%s_epoch_%i' % (savename, savename,cont_epoch))
model = tf.keras.models.load_model('saved_class/cnn_notf_withae_epoch_100')


base_dir = 'db/'
dir_det = 'DET/'
MTN = 1

images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
images_dir_loc = '/media/gsonja/Samsung_T5/ScratchDetection/MeasurementCampaigns/'

X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)
X_test_det_list = [images_dir_loc + s for s in X_test_det_list]
print(np.array(X_test_det_list)[0])
Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()

bce = tf.keras.losses.BinaryCrossentropy()
test_def_dataset = create_cnn_dataset(X_test_det_list, Y_test_det_list, _shuffle=False)

from sklearn.metrics import confusion_matrix, log_loss

def label_to_box(li):
    x = np.floor(li/24)
    y = li % 24
    return x*160, y*160

from matplotlib.patches import Rectangle
import time
import timeit
loss = tf.keras.metrics.Mean()
encode_times = []
split_times = []
predict_times = []
tot_times = []
for i in range(0,1):
    print(i)
    for i in range(0,1):
        fig, ax = plt.subplots()
        y=Y_test_det_list[i].flatten()
        y = np.array(y)
        x = np.load(X_test_det_list[i])
        ax.imshow(x)
        print(len(y))


        print(y.shape, x.shape)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        tot1 = time.time()
        img, ae_time, split_time = encode_split(ae, x)
        img = img.reshape(-1,160,160,1)
        encode_times.append(ae_time)
        split_times.append(split_time)
        t1 = time.time()
        pred = model.predict(img)
        ind = 0
        for j in pred:
            if j > 0.7:
                bx, by = label_to_box(ind)
                rect = Rectangle((by, bx), 160, 160, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            ind = ind + 1
        tot2 = time.time()
        t2 = time.time()
        tot_times.append(tot2-tot1)
        predict_times.append(t2-t1)
        plt.show()
        test_loss = log_loss(y, pred.astype("float64"), labels=[0,1])
        tn, fp, fn, tp = confusion_matrix(y, np.round(pred)).ravel()
        print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
        print('test score', test_loss)

print('Encode time ; ', np.mean(encode_times))
print('Split time ; ', np.mean(split_times))
print('Pred time ; ', np.mean(predict_times))
print('tot time ; ', np.mean(tot_times))


