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
parser.add_argument("--one_weight", type=float,
                    help="anomaly weight", default=100., required=False)
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
anomaly_weight = args.one_weight

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

if use_ae:
    ae = AutoEncoder()
    print('Loading autoencoder and data...')
    ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_TQ3_more_data/AE_TQ3_318_to_318_epochs')

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
dir_ae = 'AE/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

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

import matplotlib
from helpers.dataset_helpers import box_index_to_coords
def plot_examples(ds):
    for x, y in ds:
        x = tf.reshape(x, [2720, 3840, 1])
        fig, ax = plt.subplots()
        ax.imshow(x, vmin = 0, vmax = 255.)
        y = y.numpy().flatten()
        ind = 0
        for i in y:
            if i == 1.:
                true_x, true_y = box_index_to_coords(ind)
                rec = matplotlib.patches.Rectangle((true_x, true_y), 160, 160, facecolor='None', edgecolor='red')
                ax.add_patch(rec)
            ind = ind +1
        plt.show()

@tf.function
def encode(img, lbl, ae):
    img = tf.reshape(img, [-1, 2720, 3840, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.abs(tf.subtract(img, decoded_img))
    return aed_img, lbl

@tf.function
def format_data(image, label):
    image = tf.reshape(image, [2720, 3840, 1])
    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, [408])
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
    aed_img = tf.abs(tf.subtract(img, decoded_img))
    return aed_img, lbl

@tf.function
def process_crop_bright_encode(image_label, delta):
    image, label = image_label
    image, label = crop(image, label)
    image, label = bright_encode(image, label, ae, delta)
    return image,label

def weighted_bincrossentropy(true, pred, weight_zero=1.0, weight_one=anomaly_weight):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bin_crossentropy = bce(true, pred)

    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return tf.keras.backend.mean(weighted_bin_crossentropy)

@tf.function
def process_crop(image, label):
    image, label = crop(image, label)
    return image,label

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET_very_cleaned.npy', allow_pickle=True)
X_val_det_list = np.load(base_dir + dir_det + 'X_test_DET_very_cleaned.npy', allow_pickle=True)

Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET_very_cleaned.npy', allow_pickle=True).tolist()
Y_val_det_list = np.load(base_dir + dir_det + 'Y_test_DET_very_cleaned.npy', allow_pickle=True).tolist()

N_det_val = int(len(X_val_det_list)/2)
N_det_train = len(X_train_det_list)
X_val_det_list = X_val_det_list[:N_det_val]
Y_val_det_list = Y_val_det_list[:N_det_val]

np.random.seed(42)
X_train_norm_list = np.random.choice(np.load(base_dir + dir_ae + 'X_train_AE.npy', allow_pickle=True), N_det_train)
X_val_norm_list = np.random.choice(np.load(base_dir + dir_ae + 'X_test_AE.npy', allow_pickle=True), N_det_val)

Y_val_norm_list = np.full((N_det_val, 408), 0)
Y_train_norm_list = np.full((N_det_train, 408), 0)

add_normal = True
if add_normal == True:
    X_train_det_list = np.append(X_train_det_list, X_train_norm_list, axis = 0)
    X_val_det_list = np.append(X_val_det_list, X_val_norm_list, axis = 0)

    Y_train_det_list = np.append(Y_train_det_list, Y_train_norm_list, axis = 0)
    Y_val_det_list  = np.append(Y_val_det_list, Y_val_norm_list, axis = 0)

print('Loaded number of train, val samples: ', N_det_train, N_det_val)
print('All loaded. Starting dataset creation...')
time1 = time.time()

X_val_det_list = [images_dir_loc + s for s in X_val_det_list]
X_train_det_list = [images_dir_loc + s for s in X_train_det_list]

train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=False)
print(len(X_train_det_list), len(Y_train_det_list), np.array(Y_train_det_list).shape)
train_ds_unprocess = train_ds
val_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list, _shuffle=False)
print(len(X_val_det_list),len(Y_val_det_list), np.array(Y_val_det_list).shape)
train_ds = train_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

brightnesses = np.random.uniform(low = 0.5, high = 0.9, size = np.array(Y_train_det_list).shape[0])
counter1 = tf.data.Dataset.from_tensor_slices(brightnesses)
train_ds_counter = tf.data.Dataset.zip((train_ds_unprocess, counter1))
train_ds_brightness = train_ds_counter.map(lambda x, z: process_crop_bright_encode(x, z), num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = train_ds.concatenate(train_ds_brightness)
train_ds = train_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds_final = train_ds.cache().shuffle(buffer_size=N_det_train*2, reshuffle_each_iteration=True)
val_ds = val_ds.cache()

train_ds_examples = train_ds_final.take(10)
plot_examples(train_ds_examples)

train_ds_batch = train_ds_final.batch(batch_size=batch_size, drop_remainder = True)
val_ds_batch = val_ds.batch(batch_size=batch_size, drop_remainder = False)

time2 = time.time()
pro_time = time2-time1
print('Dataset (time {:.2f} s) created, starting training...'.format(pro_time))

optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)

if load == 'True':
    print('Loading previously trained model...')
    model = tf.keras.models.load_model('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s_epoch_%s' % (savename, savename, cont_epoch), compile=False)
    def scheduler(lr):
        return lr * tf.math.exp(-0.01)
else:
    from whole_CNNs import *
    if model_ID == 'model_whole':
        model = model_whole
    if model_ID == 'model_whole2':
        model = model_whole2
    if model_ID == 'model_whole3':
        model = model_whole3
    if model_ID == 'model_whole4':
        model = model_whole4
    def scheduler(epoch, lr):
        if epoch < 100:
            return lr
        else:
            return lr * tf.math.exp(-0.01)

print(model.summary())
model.compile(optimizer = optimizer, loss= weighted_bincrossentropy, metrics=METRICS)

filepath = 'saved_CNNs/%s/cnn_%s_epoch_{epoch:02d}' % (savename, savename)

##DO NOT SAVE EVERY EPOCH TAKES SPACE
save_every = int(np.floor((N_det_train*2)/batch_size)*10)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq=save_every)

filename = 'saved_CNNs/%s/history_log.csv' % savename
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

print('Starting training:')
history = model.fit(train_ds_batch.prefetch(tf.data.experimental.AUTOTUNE), epochs = num_epochs, validation_data = val_ds_batch.prefetch(tf.data.experimental.AUTOTUNE), callbacks = [checkpoint_callback, history_logger, lr_schedule], verbose = 1)
print('Training finished, plotting...')

plot_metrics(history)

