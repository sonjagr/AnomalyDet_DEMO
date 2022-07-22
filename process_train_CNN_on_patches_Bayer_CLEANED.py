import numpy as np
from helpers.dataset_helpers import create_cnn_dataset, process_anomalous_df_to_numpy
from helpers.cnn_helpers import rotate,rotate_1,rotate_2,rotate_3, bright, format_data, flip_h, flip_v, patch_images, flip, plot_metrics, plot_examples, crop, bright_encode, encode, tf_bayer2rgb
from old_codes.autoencoders import *
import random, time, argparse, os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
parser.add_argument("--loss", type=str,
                    help="Which loss function to use, 'fl' for focal loss or 'bce' for binary crossentropy", default="1", required=False)
parser.add_argument("--savename", type=str,
                    help="Name to save with", required=True)
parser.add_argument("--load", type=int,
                    help="Load old model or not", default = 0, required=False)
parser.add_argument("--contfromepoch", type=int,
                    help="Epoch to continue training from", default=1, required=False)
parser.add_argument("--use_ae", type=int,
                    help="use ae or not", default=1, required=False)
parser.add_argument("--gamma", type=float,
                    help="gammma for fl", default=2, required=False)
parser.add_argument("--bright_aug", type=int,
                    help="augment brightness or not", default=1, required=False)

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
loss = args.loss
gamma = args.gamma

def write_file():
    f = open('saved_CNNs/%s/argfile.txt' % savename, "w")
    f.write("  Epochs: %s" % num_epochs)
    f.write("  Batch_size: %s" % batch_size)
    f.write("  Model_ID: %s" % model_ID)
    f.write("  Savename: %s" % savename)
    f.write("  Lr: %s" % lr)
    f.write("  Loss: %s" % loss)
    f.write("  AE: %s" % use_ae)
    f.write("  Gamma: %s" % gamma)
    f.close()

if gpu != 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

try:
    os.makedirs('saved_CNNs/%s' % savename)
except FileExistsError:
    pass

random.seed(42)
base_dir = TrainDir_gpu
images_dir_loc = imgDir_gpu
dir_det = 'db/DET/'
dir_ae = 'db/AE/'

if use_ae == 1:
    ae = AutoEncoder()
    print('Loading autoencoder and data...')
    ae.load(os.path.join(base_dir, 'checkpoints/TQ3_2_1_TQ3_2_more_params_2/AE_TQ3_2_277_to_277_epochs'))

## extract normal images for training
X_list_norm = np.load(os.path.join(base_dir, 'db/NORMAL_TRAIN_20220711.npy'), allow_pickle=True)
print('        Available normal train, val images', len(X_list_norm))

f = os.path.join(base_dir, 'db/TRAIN_DATABASE_20220711')
with pd.HDFStore( f,  mode='r') as store:
        train_val_db = store.select('db')
        print(f'Reading {f}')
X_train_val_list, Y_train_val_list = process_anomalous_df_to_numpy(train_val_db)

X_train_det_list, X_val_det_list, Y_train_det_list, Y_val_det_list = train_test_split(X_train_val_list, Y_train_val_list, test_size = 70, random_state = 42)

X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_val_det_list = [images_dir_loc + s for s in X_val_det_list]

N_det_val = len(X_val_det_list)
N_det_train = len(X_train_det_list)

np.random.seed(42)

times_normal = 6
needed_nbr_of_normals = (N_det_train + N_det_val)*times_normal

X_list_norm = np.random.choice(X_list_norm, needed_nbr_of_normals, replace=False)
X_train_normal_list, X_val_normal_list = train_test_split(X_list_norm, test_size = N_det_val*times_normal, random_state = 42)

X_train_normal_list = [images_dir_loc + s for s in X_train_normal_list]
X_val_normal_list = [images_dir_loc + s for s in X_val_normal_list]

N_normal_train = len(X_train_normal_list)
N_normal_val = len(X_val_normal_list)

print('    Loaded number of defective train, val samples: ', N_det_train, N_det_val)
print('    Loaded number of normal train, val samples: ', N_normal_train, N_normal_val)
time1 = time.time()

## only images with defects
train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list.tolist(), _shuffle=False)
val_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list.tolist(), _shuffle=False)

## only normal images
normal_train_ds = create_cnn_dataset(X_train_normal_list, np.full((int(N_normal_train), PATCHES), 0.))
normal_val_ds = create_cnn_dataset(X_val_normal_list, np.full((int(N_normal_val), PATCHES), 0.))

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

if load == 1:
    print('    Loading previously trained model...')
    model = tf.keras.models.load_model(os.path.join(base_dir, 'saved_CNNs/%s/cnn_%s_epoch_%s' % (savename, savename, cont_epoch)))
if load == 0:
    print('    Not loading an old model')
    from new_CNNs import *
    if model_ID == 'vgg':
        model = VGG16()
    if model_ID == 'vgg_small':
        model = VGG16_small()
    if model_ID == 'vgg_small2':
        model = VGG16_small2()
    if model_ID == 'vgg_small_do_bn':
        model = VGG16_small_do_bn()
    if model_ID == 'vgg_small2_do':
        model = VGG16_small2_do()

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

if loss == 'bce':
    print('    Using bce')
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
elif loss == 'fl':
    print('    Using focal loss with gamma = %s' % gamma)
    loss = SigmoidFocalCrossEntropy(gamma = gamma, alpha=0.25)
model.compile(optimizer = optimizer, loss = loss, metrics=METRICS)

print(model.summary())

print('All loaded. Starting dataset processing...')

@tf.function
def process_crop_encode(image, label):
    image, label = crop(image, label)
    image, label = encode(image, label, ae)
    return image, label

@tf.function
def process_crop_bright_encode(image_label, delta):
    image, label = image_label
    image, label = crop(image, label)
    image, label = bright_encode(image, label, ae, delta)
    return image, label

@tf.function
def process_crop(image, label):
    image, label = crop(image, label)
    return image, label

@tf.function
def process_crop_bright(image, label, delta):
    image, label = crop(image, label)
    image, label = bright_encode(image, label, ae, delta)
    return image, label

def filter_anom(x):
  return tf.math.equal(x, 1)

def filter_norm(x):
  return tf.math.equal(x, 0)

np.random.seed(42)
train_ds_orig = train_ds

if use_ae == 1:
    print('    Applying autoencoding')
    print('    Using bayer format')
    train_ds = train_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    normal_train_ds = normal_train_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    normal_val_ds = normal_val_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
elif use_ae == 0:
    print('    Not applying autoencoding')
    print('    Using bayer format')
    train_ds = train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    normal_train_ds = normal_train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    normal_val_ds = normal_val_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

## add changes in brighness
np.random.seed(42)

if bright_aug == 1 and use_ae == 1:
    print('    Augmenting brightness of images')
    brightnesses = np.random.uniform(low = 0.75, high = 1.25, size = np.array(Y_train_det_list).shape[0])
    counter1 = tf.data.Dataset.from_tensor_slices(brightnesses)
    train_ds_to_bright = tf.data.Dataset.zip((train_ds_orig, counter1))
    train_ds_brightness = train_ds_to_bright.map(lambda x, z: process_crop_bright_encode(x, z), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds_brightness = train_ds_brightness.flat_map(patch_images).unbatch()

## apply patching
train_ds = train_ds.flat_map(patch_images).unbatch()
val_ds = val_ds.flat_map(patch_images).unbatch()
normal_train_ds = normal_train_ds.flat_map(patch_images).unbatch()
normal_val_ds = normal_val_ds.flat_map(patch_images).unbatch()

train_ds_anomaly = train_ds.filter(lambda x, y:  filter_anom(y))
val_ds_anomaly = val_ds.filter(lambda x, y: filter_anom(y))
plot_examples(train_ds_anomaly.skip(20).take(5))

nbr_anom_train_patches = len(list(train_ds_anomaly))
nbr_anom_val_patches = len(list(val_ds_anomaly))

if bright_aug == 1:
    train_ds_brightness_anomaly = train_ds_brightness.filter(lambda x, y: filter_anom(y))
    train_ds_anomaly = train_ds_anomaly.concatenate(train_ds_brightness_anomaly)

print('    Number of anomalous training, validation patches: ', nbr_anom_train_patches, nbr_anom_val_patches)

if bright_aug == 1:
    aug_size = 2 * nbr_anom_train_patches
if bright_aug == 0:
    aug_size = nbr_anom_train_patches

rotations = np.random.randint(low = 1, high = 4, size = aug_size).astype('int32')
counter2 = tf.data.Dataset.from_tensor_slices(rotations)
train_ds_to_rotate = tf.data.Dataset.zip((train_ds_anomaly, counter2))
train_ds_rotated = train_ds_to_rotate.map(rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)

flip_seeds = np.random.randint(low = 1, high = 11, size = aug_size).astype('int32')
counter3 = tf.data.Dataset.from_tensor_slices(flip_seeds)
train_ds_to_flip = tf.data.Dataset.zip((train_ds_anomaly, counter3))
train_ds_flipped = train_ds_to_flip.map(flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#train_ds_to_rotate = train_ds_anomaly
#rotated_1 = train_ds_to_rotate.map(rotate_1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#rotated_2 = train_ds_to_rotate.map(rotate_2, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#rotated_3 = train_ds_to_rotate.map(rotate_3, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#train_ds_rotated = train_ds_anomaly.concatenate(rotated_1).concatenate(rotated_2).concatenate(rotated_3)
#train_ds_to_flip = train_ds_anomaly
#flipped_h = train_ds_to_flip.map(flip_h, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#flipped_v = train_ds_to_flip.map(flip_v, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#train_flipped = flipped_v.concatenate(flipped_h)
#train_ds_rotated = train_ds_anomaly.concatenate(train_ds_to_rotate.map(rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE))

train_ds_rotated_flipped = train_ds_anomaly.concatenate(train_ds_rotated).concatenate(train_ds_flipped)

augmented = train_ds_rotated_flipped.cache()

anomalous_train = aug_size * 3

normal_train = N_normal_train*PATCHES
frac = int(normal_train/anomalous_train)
print('    Number of normal, anomalous training samples: ', normal_train, anomalous_train)
print('    Anomaly weight in training data: ', frac)

val_ds_final = val_ds_anomaly.concatenate(normal_val_ds.take(nbr_anom_val_patches*frac)).cache()
train_ds_final = normal_train_ds.concatenate(augmented)

train_ds_final = train_ds_final.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds_final = val_ds_final.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds_final = train_ds_final.shuffle(buffer_size=normal_train + anomalous_train, reshuffle_each_iteration=True)

train_ds_batch = train_ds_final.batch(batch_size=batch_size, drop_remainder = True)
val_ds_batch = val_ds_final.batch(batch_size=batch_size, drop_remainder = False)

plot_examples(normal_train_ds.skip(10).take(5))

time2 = time.time()
pro_time = time2-time1
print('Training and validation datasets created (processing time was {:.2f} s), starting training...'.format(pro_time))

filepath = 'saved_CNNs/%s/cnn_%s_epoch_{epoch:02d}' % (savename, savename)
steps_per_epoch = int(np.floor((normal_train+anomalous_train)/batch_size))
save_every = steps_per_epoch*2
print('    Model will be saved every %s training step, %s epoch ' % (save_every, int(save_every/steps_per_epoch)))
filepath_loss = 'saved_CNNs/%s/cnn_%s_epoch_{epoch:02d}' % (savename, savename)
checkpoint_callback_best_loss = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_loss, monitor='val_loss', mode='min', verbose=1, save_best_only=False, save_freq = save_every)

##save training history to file
filename = 'saved_CNNs/%s/history_log.csv' % savename
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

write_file()

print('Starting training:')
history = model.fit(train_ds_batch.prefetch(1), epochs = num_epochs, validation_data = val_ds_batch.prefetch(1), callbacks = [checkpoint_callback_best_loss, history_logger])
print('Training finished, plotting...')

plot_metrics(history, savename)


