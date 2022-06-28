import numpy as np
INPUT_DIM = 3
from helpers.dataset_helpers import create_cnn_dataset
from helpers.cnn_helpers import rotate, format_data, patch_images, flip, plot_metrics, plot_examples, crop, bright_encode, encode, encode_rgb, bright_encode_rgb, tf_bayer2rgb
from old_codes.autoencoders import *
import random, time, argparse, os
import tensorflow as tf
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
parser.add_argument("--savename", type=str,
                    help="Name to save with", required=True)
parser.add_argument("--load", type=str,
                    help="Load old model or not", default = False, required=True)
parser.add_argument("--contfromepoch", type=bool,
                    help="Epoch to continue training from", default=1, required=False)
parser.add_argument("--use_ae", type=str,
                    help="use ae or not", default=True, required=False)
parser.add_argument("--bright_aug", type=bool,
                    help="augment brighness or not", default=True, required=False)
parser.add_argument("--data_format", type=bool,
                    help="rgb or bayer", default='bayer', required=False)
parser.add_argument("--take_normal", type=int,
                    help="how many times normal", default=10, required=False)
parser.add_argument("--anomaly_weight", type=int,
                    help="anomaly_weight", default=2, required=False)

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
data_format = args.data_format
take_normal = args.take_normal
anomaly_weight = args.anomaly_weight

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
f.write("  Data format: %s" % data_format)
f.write("  AE: %s" % use_ae)
f.write("  Nbr of times normal: %s" % take_normal)
f.write("  Anomaly_weight: %s" % anomaly_weight)
f.close()

random.seed(42)

if use_ae:
    ae = AutoEncoder()
    print('Loading autoencoder and data...')
    ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_TQ3_more_data/AE_TQ3_401_to_401_epochs')

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
dir_ae = 'AE/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

## extract normal images for training
X_train_list_ae = np.load(os.join.path(base_dir, dir_ae, 'X_train_AE.npy'), allow_pickle=True)
X_test_list_ae = np.load(os.join.path(base_dir, dir_ae, 'X_test_AE.npy'), allow_pickle=True)

np.random.seed(1)
X_train_list_normal_to_remove = np.random.choice(X_train_list_ae, 16000, replace=False)
X_test_val_list_normal_to_remove = np.random.choice(X_test_list_ae, 4000, replace=False)

X_train_list_normal_removed = [x for x in X_train_list_ae if x not in X_train_list_normal_to_remove]
X_test_list_normal_removed = [x for x in X_test_list_ae if x not in X_test_val_list_normal_to_remove]

## load the images containing anomalies
X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET_2.npy', allow_pickle=True)
X_val_det_list = np.load(base_dir + dir_det + 'X_test_DET_2.npy', allow_pickle=True)

X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_val_det_list = [images_dir_loc + s for s in X_val_det_list]

Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET_2.npy', allow_pickle=True).tolist()
Y_val_det_list = np.load(base_dir + dir_det + 'Y_test_DET_2.npy', allow_pickle=True).tolist()

## split test and validation sets
N_det_val = int(len(X_val_det_list)/2)
N_det_train = len(X_train_det_list)
X_val_det_list = X_val_det_list[:N_det_val]
Y_val_det_list = Y_val_det_list[:N_det_val]

X_train_normal_list = np.random.choice(X_train_list_normal_removed, int(N_det_train), replace=False)
X_val_normal_list = np.random.choice(X_test_list_normal_removed, int(N_det_val), replace=False)
X_train_normal_list = [images_dir_loc + s for s in X_train_normal_list]
X_val_normal_list = [images_dir_loc + s for s in X_val_normal_list]

print('    Loaded number of train, val samples: ', N_det_train, N_det_val)
print('All loaded. Starting dataset creation...')
time1 = time.time()

## combine anomalous and normal images
X_train_list = np.append(X_train_det_list, X_train_normal_list, axis = 0)
X_val_list = np.append(X_val_det_list, X_val_normal_list, axis = 0)

shape_for_normal = list(np.shape(Y_train_det_list)[1:])
print(shape_for_normal[0])
Y_train_list = np.append(Y_train_det_list, np.full((int(len(X_train_normal_list)), shape_for_normal[0]), 0.), axis = 0)
Y_val_list = np.append(Y_val_det_list, np.full((int(len(X_val_normal_list)),  shape_for_normal[0]), 0.), axis = 0)

## only with defects
train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=False)
val_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list, _shuffle=False)

## with normal images
more_normal_train_ds = create_cnn_dataset(X_train_list, Y_train_list, _shuffle=False)
more_normal_val_ds = create_cnn_dataset(X_val_list, Y_val_list, _shuffle=False)

train_ds = more_normal_train_ds
val_ds = more_normal_val_ds

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

@tf.function
def process_crop_bright_encode(image_label, delta):
    image, label = image_label
    image, label = crop(image, label)
    image, label = bright_encode(image, label, ae, delta)
    return image, label

@tf.function
def process_crop_bright_encode_rgb(image_label, delta):
    image, label = image_label
    image, label = crop(image, label)
    image, label = bright_encode_rgb(image, label, ae, delta)
    return image, label

@tf.function
def process_crop_encode(image, label):
    image, label = crop(image, label)
    image, label = encode(image, label, ae)
    return image, label

@tf.function
def process_crop_encode_rgb(image, label):
    image, label = crop(image, label)
    image, label = encode_rgb(image, label, ae)
    return image, label

@tf.function
def process_crop(image, label):
    image, label = crop(image, label)
    return image, label

@tf.function
def process_crop_rgb(image, label):
    image, label = crop(image, label)
    image = tf_bayer2rgb(image)
    return image, label

if use_ae == True:
    print('    Applying autoencoding')
    if data_format == 'bayer':
        print('   Using bayer format')
        train_ds_nob = train_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if data_format == 'rgb':
        print('   Using rgb format')
        train_ds_nob = train_ds.map(process_crop_encode_rgb, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(process_crop_encode_rgb, num_parallel_calls=tf.data.experimental.AUTOTUNE)
if use_ae == False:
    print('    Not applying autoencoding')
    if data_format == 'bayer':
        print('   Using bayer format')
        train_ds_nob = train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if data_format == 'rgb':
        print('   Using rgb format')
        train_ds_nob = train_ds.map(process_crop_rgb, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(process_crop_rgb, num_parallel_calls=tf.data.experimental.AUTOTUNE)

## add changes in brighness
np.random.seed(42)

if bright_aug == True:
    print('    Augmenting brightness of images')
    brightnesses = np.random.uniform(low = 0.75, high = 1.25, size = np.array(Y_train_det_list).shape[0])
    counter1 = tf.data.Dataset.from_tensor_slices(brightnesses)
    train_ds_c = tf.data.Dataset.zip((train_ds, counter1))
    if data_format == 'bayer':
        train_ds_brightness = train_ds_c.map(lambda x, z: process_crop_bright_encode(x, z), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if data_format == 'rgb':
        train_ds_brightness = train_ds_c.map(lambda x, z: process_crop_bright_encode_rgb(x, z), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds_brightness = train_ds_brightness.flat_map(patch_images).unbatch()

## apply patching
train_ds = train_ds_nob.flat_map(patch_images).unbatch()
val_ds = val_ds.flat_map(patch_images).unbatch()

train_ds_anomaly = train_ds.filter(lambda x, y:  y == 1.)
nbr_anom_patches = len(list(train_ds_anomaly))

if bright_aug == True:
    train_ds_brightness_anomaly = train_ds_brightness.filter(lambda x, y:  y == 1.)
    train_ds_anomaly = train_ds_anomaly.concatenate(train_ds_brightness_anomaly)

print('    Number of anomalous patches: ', nbr_anom_patches)

if bright_aug == True:
    aug_size = 2*nbr_anom_patches
if bright_aug == False:
    aug_size = nbr_anom_patches

rotations = np.random.randint(low = 1, high = 4, size = aug_size).astype('int32')
counter2 = tf.data.Dataset.from_tensor_slices(rotations)
train_ds_to_rotate = tf.data.Dataset.zip((train_ds_anomaly, counter2))

flip_seeds = np.random.randint(low = 1, high = 11, size = aug_size).astype('int32')
counter3 = tf.data.Dataset.from_tensor_slices(flip_seeds)
train_ds_to_flip = tf.data.Dataset.zip((train_ds_anomaly, counter3))

train_ds_rotated = train_ds_anomaly.concatenate(train_ds_to_rotate.map(rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE))
train_ds_rotated_flipped = train_ds_rotated.concatenate(train_ds_to_flip.map(flip, num_parallel_calls=tf.data.experimental.AUTOTUNE))

augmented = train_ds_rotated_flipped.cache()
anomalous = nbr_anom_patches*6
normal = anomalous*take_normal

if data_format == 'bayer':
    mean_error = 26.6
if data_format == 'rgb':
    mean_error = 20.4

train_ds_normal_all = train_ds.filter(lambda x,y: y == 0.).shuffle(300000)

train_ds_normal_l = train_ds_normal_all.filter(lambda x, y: tf.reduce_mean(x) > mean_error).take(int(0.8*normal)).cache()
train_ds_normal_s = train_ds_normal_all.filter(lambda x, y: tf.reduce_mean(x) < mean_error).take(normal-int(0.8*normal)).cache()

train_ds_normal = train_ds_normal_s.concatenate(train_ds_normal_l)
normal = len(list(train_ds_normal))
train_ds_final = train_ds_normal.concatenate(augmented)

## if you want balanced recon error in normal patches
#train_ds_final = train_ds_normal_all.concatenate(augmented)

train_ds_final = train_ds_final.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

## same fraction of normals per anomalous in val data set is not necessary but can be done
#val_ds_anomaly = val_ds.filter(lambda x, y:  y == 1.)
#nbr_anom_val_patches = len(list(val_ds_anomaly))
#val_ds_normal = val_ds.filter(lambda x,y : y ==0).take(nbr_anom_val_patches*anomaly_weight)
#val_ds_final = val_ds_normal.concatenate(val_ds_anomaly)
#val_ds_final = val_ds_final.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

val_ds_final = val_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
print('    Number of normal, anomalous samples: ', normal, anomalous)
print('    Anomaly weight: ', anomaly_weight)

train_ds_final = train_ds_final.shuffle(buffer_size=normal+anomalous, reshuffle_each_iteration=True)
val_ds_final = val_ds_final.cache()

train_ds_batch = train_ds_final.batch(batch_size=batch_size, drop_remainder = True)
val_ds_batch = val_ds_final.batch(batch_size=batch_size, drop_remainder = False)

#plot_examples(train_ds_anomaly.skip(2).take(5))
#plot_examples(train_ds_normal.skip(2).take(5))
#plot_examples(val_ds.take(5))

time2 = time.time()
pro_time = time2-time1
print('Dataset (time {:.2f} s) created, starting training...'.format(pro_time))

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

if load == True:
    print('    Loading previously trained model...')
    model = tf.keras.models.load_model('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s' % (savename, savename))
    def scheduler(lr):
        return lr * tf.math.exp(-0.01)
if load == False:
    from new_CNNs import *
    if model_ID == 'model_whole':
        model = model_whole
    if model_ID == 'model_whole_final':
        model = model_whole_final
    if model_ID == 'model_whole_final2':
        model = model_whole_final2
    if model_ID == 'model_whole_final3':
        model = model_whole_final3
    if model_ID == 'model_whole_final4':
        model = model_whole_final4
    def scheduler(epoch, lr):
        if epoch < 50:
            return lr
        else:
            return lr * tf.math.exp(-0.01)

model.compile(optimizer = optimizer, loss = tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=METRICS)

print(model.summary())

class_weights = {0: 1, 1: anomaly_weight}

#filepath = 'saved_CNNs/%s/cnn_%s_epoch_{epoch:02d}' % (savename, savename)
filepath = 'saved_CNNs/%s/cnn_%s_cont' % (savename, savename)

##save every 10th epoch
save_every = int(np.floor((normal+anomalous)/batch_size)*5)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

##save training history to file
filename = 'saved_CNNs/%s/history_log_cont.csv' % savename
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

##add lr schedule
lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

print('Starting training:')
history = model.fit(train_ds_batch.prefetch(tf.data.experimental.AUTOTUNE), epochs = num_epochs, validation_data = val_ds_batch.prefetch(tf.data.experimental.AUTOTUNE), class_weight=class_weights, callbacks = [checkpoint_callback, history_logger])
print('Training finished, plotting...')

plot_metrics(history, savename)

