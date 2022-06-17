import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from helpers.dataset_helpers import create_cnn_dataset
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
import random, time, argparse
import tensorflow as tf

def plot_examples(ds):
    i = 1
    for x, y in ds:
        x = tf.reshape(x, [160, 160, 1])
        plt.imshow(x, vmin = 0, vmax = 200.)
        plt.title(str(y)+'  '+str(i))
        i = i+1
        plt.show()

@tf.function
def crop(img, lbl):
    img = tf.reshape(img, [-1, 2736, 3840, 1])
    img = tf.keras.layers.Cropping2D(cropping=((0, 16), (0, 0)))(img)
    return img, lbl

@tf.function
def encode(img, lbl, ae):
    img = tf.reshape(img, [-1, 2720, 3840, 1])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.sqrt(tf.pow(tf.subtract(img, decoded_img), 2))
    return aed_img, lbl

@tf.function
def bright_encode(img, lbl, ae):
    delta = tf.random.uniform([], minval=0.75, maxval=0.95, seed=None, dtype=tf.float32)
    img = tf.math.multiply(img, delta)
    img = tf.reshape(img, [-1, 2720, 3840, 1])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.sqrt(tf.pow(tf.subtract(img, decoded_img), 2))
    return aed_img, lbl

@tf.function
def patch_images(img, lbl):
    split_img = tf.image.extract_patches(images=img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1], rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [17*24, 160 *160])
    lbl = tf.reshape(lbl, [17*24])
    patch_ds = tf.data.Dataset.from_tensors((re, lbl))
    return patch_ds

@tf.function
def patch_image(img, lbl):
    split_img = tf.image.extract_patches(images=img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1], rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [17*24, 160 *160])
    lbl = tf.reshape(lbl, [17*24])
    return re, lbl

@tf.function
def process_crop_bright_encode(image, label):
    image, label = crop(image, label)
    image, label = bright_encode(image, label, ae)
    return image,label

@tf.function
def process_crop_encode(image, label):
    image, label = crop(image, label)
    image, label = encode(image, label, ae)
    return image,label

@tf.function
def process_crop(image, label):
    image, label = crop(image, label)
    return image,label

@tf.function
def rotate(image_label, seed):
    image, label = image_label
    image = tf.reshape(image, [1, 160, 160])
    rots = tf.random.uniform(shape=(), minval=1, maxval=4, dtype=tf.int32, seed = None)
    rot = tf.image.rot90(image, k=rots)
    return tf.reshape(rot, [160,160,1]), label

## change brightness of patch
@tf.function
def change_bright(image, label):
    delta = tf.random.uniform([], minval=0.75, maxval=0.95, seed = None, dtype=tf.float32)
    image = tf.math.multiply(image, delta)
    return image, label

@tf.function
def format(image, label):

    image = tf.reshape(image, [160, 160, 1])
    label = tf.cast(label, tf.float32)
    return image, label

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

ae = AutoEncoder()
print('Loading autoencoder and data...')
#ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')
ae.load('C:/Users/sgroenro/PycharmProjects/anomaly-detection-2/saved_class/model_AE_TQ3_500_to_500_epochs')

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
base_dir = '/db/'
dir_det = 'AE/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
images_dir_loc = 'F:/ScratchDetection/MeasurementCampaigns/'

X_train_det_list_orig = np.load(base_dir + dir_det + 'X_test_AE.npy', allow_pickle=True)
X_train_list = [images_dir_loc + s for s in X_train_det_list_orig]


N_det_train = len(X_train_list)
Y_train_list = np.full((N_det_train, 17, 24), 0.)

print('Loaded number of train: ', N_det_train)
print('All loaded. Starting dataset creation...')
time1 = time.time()

train_ds = create_cnn_dataset(X_train_list, Y_train_list, _shuffle=False)
plt.interactive(False)
train_ds = train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

count = 0
X_train_det_list_new = []
from tqdm import tqdm
for x,y in tqdm(train_ds, total = 1600):
    patches, labels = patch_image(x,y)
    labels = labels.numpy().flatten()
    patches = patches.numpy().reshape(24*17, 160*160)
    for i in range(0, len(labels)):
        label = labels[i]
        patch = patches[i, :]
        plt.imshow(patch.reshape(160,160))
        plt.show()
        ip = input('Type a if you want to add this patch to training data, press enter if not.\n')
        if ip == 'a':
            X_train_det_list_new.append(patch)
        print(len(X_train_det_list_new))
        np.save(base_dir + dir_det + 'X_test_normal_hard_patches.npy', X_train_det_list_new)
#np.save(base_dir + dir_det + 'X_test_DET_very_cleaned.npy', X_train_det_list_new)


'''
train_ds_nob = train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds_brightness = train_ds.map(process_crop_bright_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = train_ds_nob.flat_map(patch_images).unbatch()
train_ds_brightness = train_ds_brightness.flat_map(patch_images).unbatch()

nbr_of_normal = 100

train_ds_anomaly = train_ds.filter(lambda x, y:  y == 1.)
train_ds_brightness_anomaly = train_ds_brightness.filter(lambda x, y:  y == 1.)
train_ds_anomaly = train_ds_anomaly.concatenate(train_ds_brightness_anomaly)

train_ds_normal = train_ds.filter(lambda x, y:  y == 0.).shuffle(buffer_size=300000, reshuffle_each_iteration=False, seed=42).take(nbr_of_normal)

counter = tf.data.experimental.Counter()
train_ds_to_augment = tf.data.Dataset.zip((train_ds_anomaly, counter))

train_ds_rotated = train_ds_anomaly.concatenate(train_ds_to_augment.map(rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE))
augmented = train_ds_rotated

train_ds_final = train_ds_normal.concatenate(augmented)
train_ds_final = train_ds_final.map(format, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds_anomaly2 = train_ds_final.filter(lambda x, y:  y == 1.)
train_ds_normal = train_ds_final.filter(lambda x, y: y == 0.)

plot_examples(train_ds_anomaly.skip(80).take(30))
#plot_examples(train_ds_normal.skip(50).take(30))
print('done')
'''