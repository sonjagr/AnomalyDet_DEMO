import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
import cv2
from helpers.dataset_helpers import create_dataset
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
import random, time
import tensorflow  as tf
print(tf.executing_eagerly())

random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
savename = 'testing'

## autoencode and return difference
@tf.function
def encode(img):
    img = tf.reshape(img, [-1, 2720, 3840, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.sqrt(tf.pow(tf.subtract(img, decoded_img), 2))
    return aed_img

## crop bottom part
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

## flatten labels per whole image
@tf.function
def patch_labels(lbl):
    re = tf.reshape(lbl, [17*24])
    flat_ds = tf.data.Dataset.from_tensors((re))
    return flat_ds

## calculate dataset length
def ds_length(ds):
    ds = ds.as_numpy_iterator()
    ds = list(ds)
    dataset_len = len(ds)
    return dataset_len

## rotate anomalous patches
@tf.function
def rotate_image(x):
    x = tf.reshape(x, [1, 160, 160])
    rot_angle = random.choice([0, 1, 2])
    rot = tf.image.rot90(x, k=rot_angle, name=None)
    return tf.reshape(rot, [-1])

## convert rgb to bayer format
def rgb2bayer(rgb):
    (h,w) = rgb.shape[0], rgb.shape[1]
    (r,g,b) = cv2.split(rgb)
    bayer = np.empty((h, w), np.uint8)
    bayer[0::2, 0::2] = r[0::2, 0::2]
    bayer[0::2, 1::2] = g[0::2, 1::2]
    bayer[1::2, 0::2] = g[1::2, 0::2]
    bayer[1::2, 1::2] = b[1::2, 1::2]
    return bayer

## convert bayer to rgb
def bayer2rgb(bayer):
    return cv2.cvtColor(bayer.astype('uint8'), cv2.COLOR_BAYER_RG2RGB)

## change brightness of patch
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
    return tf.convert_to_tensor(final_bayer.flatten())

def tf_bright_image(image):
  im_shape = image.shape
  [image,] = tf.py_function(bright_image, [image], [tf.uint8])
  image.set_shape(im_shape)
  return image

def preprocess(imgs, lbls, to_encode = True, normal_times = 50, to_augment = False, to_brightness = False, batch_size = 256):

    cropped_imgs = imgs.map(crop)

    #bright_cropped_imgs = cropped_imgs.map(bright_image)
    if to_encode == True:
        diff_imgs = cropped_imgs.map(encode)
    if to_encode == False:
        diff_imgs = cropped_imgs

    patched_imgs = diff_imgs.flat_map(patch_images)
    patched_lbls = lbls.flat_map(patch_labels)

    patched_imgs = patched_imgs.unbatch()
    patched_lbls = patched_lbls.unbatch()

    dataset = tf.data.Dataset.zip((patched_imgs, patched_lbls))

    anomalous_dataset = dataset.filter(lambda x, y: y == 1.)

    anomalous_images = anomalous_dataset.map(lambda x, y: x)
    anomalous_labels = anomalous_dataset.map(lambda x, y: y)
    if to_augment == True:
        rotated_dataset = anomalous_images.map(lambda x: rotate_image(x))
        rotated_anomalous_dataset = tf.data.Dataset.zip((rotated_dataset, anomalous_labels))
        combined_anomalous_dataset = anomalous_dataset.concatenate(rotated_anomalous_dataset)
        if to_brightness == True:
            bright_dataset = anomalous_images
            #rgb_dataset = bright_dataset.map(lambda x: tf_bayer2rgb(x))
            bright_dataset = bright_dataset.map(lambda x: tf_bright_image(x))
            bright_anomalous_dataset = tf.data.Dataset.zip((bright_dataset, anomalous_labels))
            bright_anomalous_dataset = bright_anomalous_dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
            combined_anomalous_dataset = combined_anomalous_dataset.concatenate(bright_anomalous_dataset)
    if to_augment == False:
        combined_anomalous_dataset = anomalous_dataset
    anomalous_nbr = ds_length(combined_anomalous_dataset)
    normal_dataset = dataset.filter(lambda x, y: y == 0.).shuffle(500).take(normal_times * anomalous_nbr)

    nbr_anomalous =  ds_length(combined_anomalous_dataset)

    combined_dataset_batch = normal_dataset.concatenate(combined_anomalous_dataset).shuffle(20000).batch(batch_size=batch_size, drop_remainder=True)
    return combined_dataset_batch, nbr_anomalous

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def dyn_weighted_bincrossentropy(true, pred):
    # get the total number of inputs
    num_pred = tf.keras.backend.sum(tf.keras.backend.cast(pred < 0.5, true.dtype)) + tf.keras.backend.sum(true)
    # get weight of values in 'pos' category
    zero_weight = tf.keras.backend.sum(true) / num_pred + tf.keras.backend.epsilon()
    # get weight of values in 'false' category
    one_weight = tf.keras.backend.sum(tf.keras.backend.cast(pred < 0.5, true.dtype)) / num_pred + tf.keras.backend.epsilon()
    # calculate the weight vector
    weights = (1.0 - true) * zero_weight + true * one_weight
    # calculate the binary cross entropy
    bin_crossentropy = bce(true, pred)
    # apply the weights
    weighted_bin_crossentropy = weights * bin_crossentropy
    return tf.keras.backend.mean(weighted_bin_crossentropy)

def loss(model, x, y, training):
    y_ = model(x, training=training)
    return dyn_weighted_bincrossentropy(true=y, pred=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

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
Y_test_det_list = Y_test_det_list[:10]

N_det_test = int(len(X_test_det_list)/2)
N_det_train = len(X_train_det_list)
print('Loaded number of train, test samples: ', N_det_train, N_det_test)
print('All loaded. Starting processing...')
time1 = time.time()

train_imgs = create_dataset(X_train_det_list)
train_lbls = tf.data.Dataset.from_tensor_slices(Y_train_det_list)

test_imgs = create_dataset(X_test_det_list)
test_lbls = tf.data.Dataset.from_tensor_slices(Y_test_det_list)

batch_size = 2

train_combined_dataset_batch, train_ano = preprocess(train_imgs, train_lbls, to_encode = True, normal_times=50, to_augment= True, to_brightness=True, batch_size = batch_size)
val_combined_dataset, val_ano = preprocess(test_imgs, test_lbls, to_encode = True, normal_times=50, to_augment= False, batch_size = 1)
val_combined_dataset = val_combined_dataset.unbatch()

train_dataset_len = ds_length(train_combined_dataset_batch.unbatch())
val_dataset_len = ds_length(val_combined_dataset)
print('Number of training, validation patches in total: ', train_dataset_len, val_dataset_len)
print('Number of anomalous training, validation patches: ', train_ano, val_ano)

time2 = time.time()
pro_time = time2-time1
print('Processing (time {:.2f} s) finished, starting training...'.format(pro_time))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

load = False
cont_epoch = 10
if load == 'True':
    print('Loading model...')
    model = tf.keras.models.load_model('saved_class/%s/cnn_%s_epoch_%i' % (savename, savename,cont_epoch))
    cont_epoch = cont_epoch
    train_loss_results = np.load('saved_CNNs/%s/cnn_%s_train_loss.npy' % (savename, savename))
    val_loss_results = np.load('saved_CNNs/%s/cnn_%s_val_loss.npy' % (savename, savename))
else:
    cont_epoch = 0
    train_loss_results, val_loss_results = [], []
    train_accuracy_results, val_accuracy_results = [], []
    from CNNs import *
    model = model_works_newdatasplit4

print(model.summary())

num_epochs = 200

for epoch in tqdm(range(cont_epoch, num_epochs), total = num_epochs):
    epoch_train_loss_mean, epoch_val_loss_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
    epoch_train_accuracy, epoch_val_accuracy = tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.BinaryAccuracy()
    #epoch_train_accuracy, epoch_val_accuracy = tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalseNegatives()

    for x, y in train_combined_dataset_batch:
        x = tf.reshape(x, [-1, 160, 160, 1])
        y = tf.cast(y, tf.float32)

        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_train_loss_mean.update_state(loss_value)
        epoch_train_accuracy.update_state(y, model(x, training=True))


    train_loss_results.append(epoch_train_loss_mean.result())
    train_accuracy_results.append(epoch_train_accuracy.result())

    for val_x, val_y in val_combined_dataset:
        val_x = tf.reshape(val_x, [-1, 160, 160, 1])
        val_y = tf.cast(val_y, tf.float32)

        val_loss_value = loss(model, val_x, val_y, training=False)
        epoch_val_loss_mean.update_state(val_loss_value)
        epoch_val_accuracy.update_state(val_y, model(val_x, training=False))

    val_loss_results.append(epoch_val_loss_mean.result())
    val_accuracy_results.append(epoch_val_accuracy.result())

    print(epoch_train_loss_mean.result(), epoch_train_accuracy.result(), epoch_val_loss_mean.result(), epoch_val_accuracy.result())
    if epoch % 5 == 0:
        print("Epoch {:03d}: Train loss: {:.10f}, train accuracy: {:.3%}. Validation loss: {:.10f}, validation accuracy: {:.3%}".format(epoch, epoch_train_loss_mean.result(), epoch_train_accuracy.result(), epoch_val_loss_mean.result(), epoch_val_accuracy.result()))
        try:
            os.makedirs('saved_cNNs/%s' % savename)
        except FileExistsError:
            pass
        model_saveto = 'saved_CNNs/%s/cnn_%s_epoch_%i' % (savename, savename, epoch)
        model.save(model_saveto)
        print('Model checkpoint saved to ', model_saveto)
        np.save('saved_CNNs/%s/cnn_%s_val_loss.npy' % (savename, savename), val_loss_results)
        np.save('saved_CNNs/%s/cnn_%s_train_loss.npy' % (savename, savename), train_loss_results)
        print('Model test and train losses saved to ', 'saved_CNNs/%s/' % savename)

        plt.plot(np.arange(0, len(val_loss_results)), val_loss_results, label = 'Val loss')
        plt.plot(np.arange(0, len(train_loss_results)), train_loss_results, label ='Train loss')
        plt.legend()
        plt.show()
