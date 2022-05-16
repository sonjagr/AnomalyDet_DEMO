import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
import cv2
from helpers.dataset_helpers import create_dataset
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
from helpers.dataset_helpers import rgb2bayer, bayer2rgb
import random
random.seed(42)

## autoencode and return difference
@tf.function
def encode(img):
    img = tf.reshape(img, [-1, 2720, 3840, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.sqrt(tf.pow(tf.subtract(img, decoded_img), 2))
    return aed_img

## crop bottom part away
@tf.function
def crop(img):
    output = tf.image.crop_to_bounding_box(image = img, offset_height = 0, offset_width = 0, target_height = 2720, target_width= 3840)
    return output

# split into patches
@tf.function
def patch_images(image):
    split_img = tf.image.extract_patches(images=image, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1], rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [17*24, 160 * 160])
    noisy_ds = tf.data.Dataset.from_tensors((re))
    return noisy_ds

@tf.function
def patch_labels(lbl):
    re = tf.reshape(lbl, [17*24])
    flat_ds = tf.data.Dataset.from_tensors((re))
    return flat_ds

def ds_length(ds):
    ds = ds.as_numpy_iterator()
    ds = list(ds)
    dataset_len = len(ds)
    return dataset_len

@tf.function
def rotate_image(x):
    x = tf.reshape(x, [1, 160, 160])
    rot_angle = random.choice([0, 1, 2])
    rot = tf.image.rot90(x, k=rot_angle, name=None)
    return tf.reshape(rot, [-1])

@tf.function
def bright_image(img):
    value = random.choice(np.arange(-51,51,1))
    value = value.astype('uint8')
    img = img.numpy()
    rgb = bayer2rgb(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    final_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    final_bayer = rgb2bayer(final_rgb)
    return tf.convert_to_tensor(final_bayer.reshape(-1, 2736, 3840, 1))

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

ae = AutoEncoder()
print('Loading autoencoder and data...')
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')
base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'
X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)
X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_test_det_list = [images_dir_loc + s for s in X_test_det_list]
Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET.npy', allow_pickle=True).tolist()
Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()

## how many samples used
X_train_det_list = X_train_det_list[:10]
Y_train_det_list = Y_train_det_list[:10]
X_test_det_list = X_test_det_list[:10]

N_det_test = int(len(X_test_det_list)/2)
N_det_train = len(X_train_det_list)
print('Loaded number of train, test samples: ', N_det_train, N_det_test)

print('All loaded. Starting processing...')

imgs = create_dataset(X_train_det_list)
lbls = tf.data.Dataset.from_tensor_slices(Y_train_det_list)

cropped_imgs = imgs.map(crop)

#bright_cropped_imgs = cropped_imgs.map(bright_image)

diff_imgs = cropped_imgs.map(encode)

patched_imgs = diff_imgs.flat_map(patch_images)
patched_lbls = lbls.flat_map(patch_labels)

patched_imgs = patched_imgs.unbatch()
patched_lbls = patched_lbls.unbatch()

dataset = tf.data.Dataset.zip((patched_imgs, patched_lbls))

anomalous_dataset = dataset.filter(lambda x,y: y == 1.)
anomalous_nbr = ds_length(anomalous_dataset)
normal_dataset = dataset.filter(lambda x,y: y == 0.).shuffle(500).take(50*anomalous_nbr)

anomalous_images = anomalous_dataset.map(lambda x, y: x)
anomalous_labels = anomalous_dataset.map(lambda x, y: y)
rotated_dataset = anomalous_images.map(lambda x: rotate_image(x))

rotated_anomalous_dataset = tf.data.Dataset.zip((rotated_dataset, anomalous_labels))
combined_anomalous_dataset = anomalous_dataset.concatenate(rotated_anomalous_dataset)

#combined_dataset = normal_dataset.concatenate(rotated_anomalous_dataset).shuffle(10000).batch(128)
combined_dataset = normal_dataset.concatenate(combined_anomalous_dataset)
combined_dataset_batch = normal_dataset.concatenate(combined_anomalous_dataset).shuffle(20000).batch(256)

dataset_len = ds_length(combined_dataset)
print('Number of training patches in total: ', dataset_len)
print('Number of anomalous patches: ', ds_length(combined_anomalous_dataset))

print('Processing finished, starting training...')

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def weighted_bce(y_true, y_pred):
  weights = (y_true.numpy() * 10.) + 1
  loss_bce = bce(y_true.numpy(), y_pred.numpy())
  weighted_bce = tf.keras.metrics.Mean(loss_bce * weights)
  return weighted_bce

def loss(model, x, y, training):
  y_ = model(x, training=training)
  y_ = tf.reshape(y_, [-1])
  print(y_.shape)
  return weighted_bce(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss_results = []
train_accuracy_results = []

num_epochs = 200

from CNNs import *
model = model_works_newdatasplit4

for epoch in tqdm(range(num_epochs), total = num_epochs):
    epoch_loss_mean = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.BinaryAccuracy()

    for x, y in combined_dataset_batch:
        x = tf.reshape(x, [-1, 160, 160, 1])
        y = tf.reshape(y, [-1])
        print(y.shape)
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_mean.update_state(loss_value)
        epoch_accuracy.update_state(y, model(x, training=True))

    train_loss_results.append(epoch_loss_mean.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 10 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,epoch_loss_mean.result(), epoch_accuracy.result()))
