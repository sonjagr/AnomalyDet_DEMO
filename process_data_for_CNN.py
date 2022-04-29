import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
from helpers.dataset_helpers import create_cnn_dataset
from autoencoders import *
from common import *
from scipy.ndimage.interpolation import rotate
import random
import matplotlib.pyplot as plt
random.seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def rotate_img(img):
    img = img.reshape(160,160)
    rot_angle = random.choice([90, 180, 270])
    rot = rotate(img, rot_angle)
    return rot.flatten()

def change_bright_img(img):
    img = img.flatten()
    change = 0.5
    img = np.multiply(img, change)
    return img

ae = AutoEncoder()
print('Loading autoencoder...')
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')

def encode_split(ae, img):
    img = img[0].numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = np.sqrt((decoded_img - img)**2)

    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    #ax1.imshow(img.reshape(2720, 3840))
    #ax2.imshow(decoded_img.numpy().reshape(2720, 3840))
    #ax3.imshow(aed_img.reshape(2720, 3840))
    #plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/testing/aed_test2.png')

    split_img = tf.image.extract_patches(images=aed_img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1],rates=[1, 1, 1, 1], padding='VALID')
    split_img = split_img.numpy().reshape(17 * 24, 160 * 160)

    #fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(1, 10)
    #ax2.imshow(split_img[(17*24)-10].reshape(160, 160))
    #ax3.imshow(split_img[(17*24)-9].reshape(160, 160))
    #ax4.imshow(split_img[(17*24)-8].reshape(160, 160))
    #ax5.imshow(split_img[(17*24)-7].reshape(160, 160))
    #ax6.imshow(split_img[(17*24)-6].reshape(160, 160))
    #ax7.imshow(split_img[(17*24)-5].reshape(160, 160))
    #ax8.imshow(split_img[(17*24)-4].reshape(160, 160))
    #ax9.imshow(split_img[(17*24)-3].reshape(160, 160))
    #ax10.imshow(split_img[(17*24)-2].reshape(160, 160))
    #ax10.imshow(split_img[(17*24)-1].reshape(160, 160))
    #plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/testing/split_test2.png')

    return split_img

def only_split(img):
    img = img[0].numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
    split_img = tf.image.extract_patches(images=img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1],rates=[1, 1, 1, 1], padding='VALID')
    return split_img.numpy().reshape(17 * 24, 160 * 160)

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)

X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_test_det_list = [images_dir_loc + s for s in X_test_det_list]

Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET.npy', allow_pickle=True).tolist()
Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()

N_det_test = int(len(X_test_det_list)/2)
N_det_train = len(X_train_det_list)
print('N train, N test : ', N_det_train, N_det_test)

train_def_dataset = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=False)
test_def_dataset = create_cnn_dataset(X_test_det_list[:N_det_test], Y_test_det_list[:N_det_test], _shuffle=False)
val_def_dataset = create_cnn_dataset(X_test_det_list[-N_det_test:], Y_test_det_list[-N_det_test:], _shuffle=False)

#how many normal images for each defective
MTN = 1

def process_train_data(dataset, MTN, seed):
    random.seed(seed)
    np.random.seed(seed)
    defects, plot = 0, 0
    lbl_list, img_list = [], []
    print('Processing '+str(dataset))
    for X, Y in tqdm(dataset, total=tf.data.experimental.cardinality(dataset).numpy()):
        Y = Y.numpy().reshape(17*24)
        i_a = np.where(Y == 1)[0]
        i_n = np.where(Y == 0)[0]
        split_img = only_split(X)
        defects = defects + len(i_a)
        if len(i_a) > 0:
            def_imgs = split_img[i_a, :]

            #if len(i_a) == 3:
            #    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            #    ax1.imshow(def_imgs[0].reshape(160, 160))
            #    ax2.imshow(def_imgs[1].reshape(160, 160))
            #    ax3.imshow(def_imgs[2].reshape(160, 160))
            #    plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/testing/split_test3.png')

            rotated = np.array([rotate_img(item) for item in def_imgs])
            def_imgs = np.append(def_imgs, rotated, axis=0)
            n_normal = def_imgs.shape[0]
            i_n = np.random.choice(a=i_n, size=n_normal * MTN, replace=False)
            norm_imgs = split_img[i_n, :]
            lbls = np.append(np.full(len(def_imgs), 1), np.full(len(norm_imgs), 0), axis=0)
            imgs = np.append(def_imgs, norm_imgs, axis=0).reshape(-1, 160, 160)
            lbl_list = np.append(lbl_list, lbls)
            img_list = np.append(img_list, imgs).reshape(-1, 160, 160)
    print('Data shape: ', img_list.shape, lbl_list.shape)
    print('Number of defective sub-images: ', defects)
    dataset_epoch = tf.data.Dataset.from_tensor_slices((img_list, lbl_list))
    return dataset_epoch, img_list, lbl_list

def process_test_val_data(dataset, MTN, seed):
    random.seed(seed)
    np.random.seed(seed)
    defects, plot = 0, 0
    lbl_list, img_list = [], []
    print('Processing '+str(dataset))
    for X, Y in tqdm(dataset, total=tf.data.experimental.cardinality(dataset).numpy()):
        Y = Y.numpy().reshape(17*24)
        i_a = np.where(Y == 1)[0]
        i_n = np.where(Y == 0)[0]
        split_img = encode_split(ae, X)
        defects = defects + len(i_a)
        if len(i_a) > 0:
            def_imgs = split_img[i_a, :]
            n_normal = def_imgs.shape[0]
            i_n = np.random.choice(a=i_n, size=n_normal * MTN, replace=False)
            norm_imgs = split_img[i_n, :]
            lbls = np.append(np.full(len(def_imgs), 1), np.full(len(norm_imgs), 0), axis=0)
            imgs = np.append(def_imgs, norm_imgs, axis=0).reshape(-1, 160, 160)
            lbl_list = np.append(lbl_list, lbls)
            img_list = np.append(img_list, imgs).reshape(-1, 160, 160)
    print('Data shape: ', img_list.shape, lbl_list.shape)
    print('Number of defective sub-images: ', defects)
    dataset_epoch = tf.data.Dataset.from_tensor_slices((img_list, lbl_list))
    return dataset_epoch, img_list, lbl_list

_, train_img_list, train_lbl_list = process_test_val_data(train_def_dataset, MTN, seed=1)
np.save('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'train_img_list_noaug_%i.npy' % MTN, train_img_list)
np.save('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'train_lbl_list_noaug_%i.npy' % MTN, train_lbl_list)

#_, test_img_list, test_lbl_list = process_test_val_data(test_def_dataset, MTN, seed = 2)
#np.save('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'test_img_list_noae_%i.npy' % MTN, test_img_list)
#np.save('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'test_lbl_list_noae_%i.npy' % MTN, test_lbl_list)

#_, val_img_list, val_lbl_list = process_test_val_data(val_def_dataset, MTN, seed = 3)
#np.save('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'val_img_list_noae_%i.npy' % MTN, val_img_list)
#np.save('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'val_lbl_list_noae_%i.npy' % MTN, val_lbl_list)

print('All done!')