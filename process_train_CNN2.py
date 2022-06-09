import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from helpers.dataset_helpers import create_cnn_dataset
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
import random, time, argparse
import tensorflow as tf

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--model_ID", type=str, help="Model ID",
                    default="model", required=True)
parser.add_argument("--batch_size", type=int,
                    help="Batch size for the training", default=256, required=False)
parser.add_argument("--epochs", type=int,
                    help="Number of epochs for the training", default=50, required=False)
parser.add_argument("--lr", type=float,
                    help="Learning rate", default=1e-3, required=False)
parser.add_argument("--gpu", type=str,
                    help="Which gpu to use", default="1", required=False)
parser.add_argument("--savename", type=str,
                    help="Name to save with", required=True)
parser.add_argument("--load", type=str,
                    help="Load old model or not", default = "False", required=True)
parser.add_argument("--contfromepoch", type=int,
                    help="Epoch to continue training from", default=1, required=False)
parser.add_argument("--use_ae", type=bool,
                    help="use ae or not", default=True, required=False)
parser.add_argument("--bright_aug", type=bool,
                    help="augment brighness or not", default=True, required=False)


args = parser.parse_args()

num_epochs = args.epochs
batch_size = args.batch_size
model_ID = args.model_ID
gpu = args.gpu
savename = args.savename
load = args.load
lr = args.lr
cont_epoch = args.contfromepoch
use_ae = args.use_ae
bright_aug = args.bright_aug

if gpu != 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

try:
    os.makedirs('saved_CNNs/%s' % savename)
except FileExistsError:
    pass

f = open('saved_CNNs/%s/argfile.txt' % savename, "w")
f.write("  Epochs: %s" % num_epochs)
f.write("  Batch_size: %s" % batch_size)
f.write("  Model_ID: %s" % model_ID)
f.write("  Savename: %s" % savename)
f.write("  Lr: %s" % lr)
f.close()

random.seed(42)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

if use_ae:
    ae = AutoEncoder()
    print('Loading autoencoder and data...')
    ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
dir_ae = 'AE/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

X_train_list_ae = np.load(base_dir + dir_ae + 'X_train_AE.npy', allow_pickle=True)
X_test_list_ae = np.load(base_dir + dir_ae + 'X_test_AE.npy', allow_pickle=True)

np.random.seed(1)
X_train_list_normal_to_remove = np.random.choice(X_train_list_ae, 8000, replace=False)
X_test_val_list_normal_to_remove = np.random.choice(X_test_list_ae, 2000, replace=False)

X_train_list_normal_removed = [x for x in X_train_list_ae if x not in X_train_list_normal_to_remove]
X_test_list_normal_removed = [x for x in X_test_list_ae if x not in X_test_val_list_normal_to_remove]

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET_very_cleaned.npy', allow_pickle=True)
X_val_det_list = np.load(base_dir + dir_det + 'X_test_DET_very_cleaned.npy', allow_pickle=True)

Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET_very_cleaned.npy', allow_pickle=True).tolist()
Y_val_det_list = np.load(base_dir + dir_det + 'Y_test_DET_very_cleaned.npy', allow_pickle=True).tolist()

N_det_val = int(len(X_val_det_list)/2)
N_det_train = len(X_train_det_list)
X_val_det_list = X_val_det_list[:N_det_val]
Y_val_det_list = Y_val_det_list[:N_det_val]

X_train_normal_list = np.random.choice(X_train_list_normal_removed, int(N_det_train), replace=False)
X_val_normal_list = np.random.choice(X_test_list_normal_removed, int(N_det_val), replace=False)

print('Loaded number of train, val samples: ', N_det_train, N_det_val)
print('All loaded. Starting dataset creation...')
time1 = time.time()

X_train_list = np.append(X_train_det_list, X_train_normal_list)
X_val_list = np.append(X_val_det_list, X_val_normal_list)
X_train_list = [images_dir_loc + s for s in X_train_list]
X_val_list = [images_dir_loc + s for s in X_val_list]

Y_train_list = np.append(Y_train_det_list, np.full((int(len(Y_train_det_list)), 408), 0.), axis =0)
Y_val_list = np.append(Y_val_det_list, np.full((int(len(Y_val_det_list)), 408), 0.), axis =0)

train_ds = create_cnn_dataset(X_train_list, Y_train_list, _shuffle=False)
val_ds = create_cnn_dataset(X_val_list, Y_val_list, _shuffle=False)

normal_train_ds = create_cnn_dataset(X_train_normal_list, np.full((len(X_train_normal_list), 408), 0.), _shuffle=False)

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'),
]

def plot_metrics(history):
    colors = ['blue','red']
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig('saved_CNNs/%s/training.png' % savename, dpi = 600)
    plt.show()

def plot_examples(ds):
    for x, y in ds:
        x = tf.reshape(x, [160, 160, 1])
        plt.imshow(x, vmin = 0, vmax = 200.)
        plt.title(str(y))
        plt.show()

@tf.function
def encode(img, lbl, ae):
    img = tf.reshape(img, [-1, 2720, 3840, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.sqrt(tf.pow(tf.subtract(img, decoded_img), 2))
    return aed_img, lbl

@tf.function
def format_data(image, label):
    image = tf.reshape(image, [160, 160, 1])
    label = tf.cast(label, tf.float32)
    return image, label

@tf.function
def crop(img, lbl):
    img = tf.reshape(img, [-1, 2736, 3840, INPUT_DIM])
    img = tf.keras.layers.Cropping2D(cropping=((0, 16), (0, 0)))(img)
    return img, lbl

@tf.function
def process_crop_encode(image, label):
    image, label = crop(image, label)
    image, label = encode(image, label, ae)
    return image,label

@tf.function
def bright_encode(img, lbl, ae, delta):
    img = tf.cast(img, tf.float64)
    img = tf.math.multiply(img, delta)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, [-1, 2720, 3840, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.sqrt(tf.pow(tf.subtract(img, decoded_img), 2))
    return aed_img, lbl

@tf.function
def process_crop_bright_encode(image_label, delta):
    image, label = image_label
    image, label = crop(image, label)
    image, label = bright_encode(image, label, ae, delta)
    return image,label

@tf.function
def rotate(image_label, rots):
    image, label = image_label
    image = tf.reshape(image, [1, 160, 160])
    rot = tf.image.rot90(image, k=rots)
    return tf.reshape(rot, [160,160,1]), label

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
def patch_images(img, lbl):
    split_img = tf.image.extract_patches(images=img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1], rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [17*24, 160 *160])
    lbl = tf.reshape(lbl, [17*24])
    patch_ds = tf.data.Dataset.from_tensors((re, lbl))
    return patch_ds

if use_ae:
    train_ds_nob = train_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
if not use_ae:
    train_ds_nob = train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

## add changes in brighness
np.random.seed(42)

if bright_aug:
    brightnesses = np.random.uniform(low = 0.5, high = 0.9, size = np.array(Y_train_det_list).shape[0])
    counter1 = tf.data.Dataset.from_tensor_slices(brightnesses)
    train_ds_c = tf.data.Dataset.zip((train_ds, counter1))
    train_ds_brightness = train_ds_c.map(lambda x,z: process_crop_bright_encode(x,z), num_parallel_calls=tf.data.experimental.AUTOTUNE)

if use_ae:
    val_ds = val_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
if not use_ae:
    val_ds = val_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

## apply patching
train_ds = train_ds_nob.flat_map(patch_images).unbatch()
plot_examples(train_ds.take(2))

if bright_aug:
    train_ds_brightness = train_ds_brightness.flat_map(patch_images).unbatch()
    plot_examples(train_ds_brightness.take(2))

val_ds = val_ds.flat_map(patch_images).unbatch()

train_ds_anomaly = train_ds.filter(lambda x, y:  y == 1.)

if bright_aug:
    train_ds_brightness_anomaly = train_ds_brightness.filter(lambda x, y:  y == 1.)
    train_ds_anomaly = train_ds_anomaly.concatenate(train_ds_brightness_anomaly)

nbr_anom_patches = 1283
nbr_anom_patches = len(list(train_ds.filter(lambda x, y:  y == 1.)))
print('Number of anom patches: ', nbr_anom_patches)

if bright_aug:
    rot_size = 2*nbr_anom_patches
if not bright_aug:
    rot_size = nbr_anom_patches
rotations = np.random.randint(low = 1, high = 4, size = rot_size).astype('int32')
counter2 = tf.data.Dataset.from_tensor_slices(rotations)
train_ds_to_augment = tf.data.Dataset.zip((train_ds_anomaly, counter2))

train_ds_rotated = train_ds_anomaly.concatenate(train_ds_to_augment.map(rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE))

augmented = train_ds_rotated

train_ds_normal = train_ds.filter(lambda x,y: y ==0.)

train_ds_final = train_ds_normal.concatenate(augmented)
train_ds_final = train_ds_final.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds_anomaly = train_ds_final.filter(lambda x, y:  y == 1.)

normal, anomalous = len(Y_train_list)*408-nbr_anom_patches, 4*nbr_anom_patches
anomaly_weight = normal/anomalous
print('Number of normal, anomalous samples: ', normal, anomalous)
print('Anomaly weight: ', anomaly_weight)

train_ds_final = train_ds_final.cache().shuffle(buffer_size=normal+anomalous, reshuffle_each_iteration=True)
val_ds = val_ds.cache()

train_ds_batch = train_ds_final.batch(batch_size=batch_size, drop_remainder = True)
val_ds_batch = val_ds.batch(batch_size=batch_size, drop_remainder = False)

plot_examples(train_ds_anomaly.take(2))
plot_examples(train_ds_normal.take(2))

time2 = time.time()
pro_time = time2-time1
print('Dataset (time {:.2f} s) created, starting training...'.format(pro_time))

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

if load == 'True':
    print('Loading previously trained model...')
    model = tf.keras.models.load_model('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s_epoch_%s' % (savename, savename, cont_epoch))
    def scheduler(lr):
        return lr * tf.math.exp(-0.01)
else:
    from new_CNNs import *
    if model_ID == 'model_tf':
        model = model_tf
    if model_ID == 'model_2':
        model = model_2
    if model_ID == 'model_3':
        model = model_3
    if model_ID == 'model_4':
        model = model_4
    def scheduler(epoch, lr):
        if epoch < 50:
            return lr
        else:
            return lr * tf.math.exp(-0.01)

print(model.summary())
model.compile(optimizer = optimizer, loss= tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=METRICS)

class_weights = {0: 1., 1: anomaly_weight}

filepath = 'saved_CNNs/%s/cnn_%s_epoch_{epoch:02d}' % (savename, savename)

##DO NOTNSAVE EVERY EPOCH TAKES SPACE
save_every = int(np.floor((normal+anomalous)/batch_size)*10)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq=save_every)

filename = 'saved_CNNs/%s/history_log.csv' % savename
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=load)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

print('Starting training:')
history = model.fit(train_ds_batch.prefetch(tf.data.experimental.AUTOTUNE), epochs = num_epochs, validation_data = val_ds_batch.prefetch(tf.data.experimental.AUTOTUNE), class_weight=class_weights, callbacks = [checkpoint_callback, history_logger])
print('Training finished, plotting...')

plot_metrics(history)

