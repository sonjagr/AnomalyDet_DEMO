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

if use_ae:
    ae = AutoEncoder()
    print('Loading autoencoder and data...')
    ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')

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
    aed_img = tf.sqrt(tf.pow(tf.subtract(img, decoded_img), 2))
    return aed_img, lbl

@tf.function
def process_crop_bright_encode(image_label, delta):
    image, label = image_label
    image, label = crop(image, label)
    image, label = bright_encode(image, label, ae, delta)
    return image,label

def weighted_bincrossentropy(true, pred, weight_zero=1.0, weight_one=200.):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bin_crossentropy = bce(true, pred)

    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return tf.keras.backend.mean(weighted_bin_crossentropy)

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

X_train_list_ae = np.load(base_dir + dir_ae + 'X_train_AE.npy', allow_pickle=True)
X_test_list_ae = np.load(base_dir + dir_ae + 'X_test_AE.npy', allow_pickle=True)

np.random.seed(1)
X_train_list_normal_to_remove = np.random.choice(X_train_list_ae, 8000, replace=False)
X_test_val_list_normal_to_remove = np.random.choice(X_test_list_ae, 2000, replace=False)

X_train_list_normal_removed = [x for x in X_train_list_ae if x not in X_train_list_normal_to_remove]
X_test_list_normal_removed = [x for x in X_test_list_ae if x not in X_test_val_list_normal_to_remove]

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_val_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)

Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET.npy', allow_pickle=True).tolist()
Y_val_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()

N_det_val = int(len(X_val_det_list)/2)
N_det_train = len(X_train_det_list)
X_val_det_list = X_val_det_list[:N_det_val]
Y_val_det_list = Y_val_det_list[:N_det_val]

X_train_normal_list = np.random.choice(X_train_list_normal_removed, int(N_det_train), replace=False)
X_val_normal_list = np.random.choice(X_test_list_normal_removed, int(N_det_val), replace=False)
Y_train_normal_list = np.full((N_det_train, 17, 24), 0.)
Y_val_normal_list = np.full((N_det_val, 17, 24), 0.)

print('Loaded number of train, val samples: ', N_det_train, N_det_val)
print('All loaded. Starting dataset creation...')
time1 = time.time()

X_train_list_comb = np.append(X_train_det_list, X_train_normal_list)
X_val_list_comb = np.append(X_val_det_list, X_val_normal_list)

X_val_det_list = [images_dir_loc + s for s in X_val_det_list]
X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_train_normal_list = [images_dir_loc + s for s in X_train_normal_list]

Y_train_list_comb = np.append(Y_train_det_list, Y_train_normal_list, axis =0)
Y_val_list_comb = np.append(Y_val_det_list, Y_val_normal_list, axis =0)

X_train_list_comb = [images_dir_loc + s for s in X_train_list_comb]
X_val_list_comb = [images_dir_loc + s for s in X_val_list_comb]

train_ds = create_cnn_dataset(X_train_list_comb, Y_train_list_comb, _shuffle=False)
val_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list, _shuffle=False)

anomalous_train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=False)
normal_train_ds = create_cnn_dataset(X_train_normal_list, Y_train_normal_list, _shuffle=False)


if use_ae:
    normal_train_ds = normal_train_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    anomalous_train_ds = anomalous_train_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
if not use_ae:
    normal_train_ds = normal_train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    anomalous_train_ds = anomalous_train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

normal_train_ds = normal_train_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
anomalous_train_ds = anomalous_train_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#train_ds_final = normal_train_ds.concatenate(anomalous_train_ds)
train_ds_final = anomalous_train_ds
val_ds = val_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

normal, anomalous = len(list(normal_train_ds)), len(list(anomalous_train_ds))
anomaly_weight = normal/anomalous
print('Number of normal, anomalous samples: ', normal, anomalous)
print('Anomaly weight: ', anomaly_weight)

train_ds_final = train_ds_final.cache().shuffle(buffer_size=normal+anomalous, reshuffle_each_iteration=True)
val_ds = val_ds.cache()

train_ds_batch = train_ds_final.batch(batch_size=batch_size, drop_remainder = True)
val_ds_batch = val_ds.batch(batch_size=batch_size, drop_remainder = False)

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
    from whole_CNNs import *
    if model_ID == 'model_whole':
        model = model_whole
    if model_ID == 'model_whole2':
        model = model_whole2
    def scheduler(epoch, lr):
        if epoch < 50:
            return lr
        else:
            return lr * tf.math.exp(-0.01)

print(model.summary())
model.compile(optimizer = optimizer, loss= weighted_bincrossentropy, metrics=METRICS)

filepath = 'saved_CNNs/%s/cnn_%s_epoch_{epoch:02d}' % (savename, savename)

##DO NOTNSAVE EVERY EPOCH TAKES SPACE
save_every = int(np.floor((anomalous)/batch_size)*10)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq=save_every)

filename = 'saved_CNNs/%s/history_log.csv' % savename
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=load)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

print('Starting training:')
history = model.fit(train_ds_batch.prefetch(tf.data.experimental.AUTOTUNE), epochs = num_epochs, validation_data = val_ds_batch.prefetch(tf.data.experimental.AUTOTUNE),  callbacks = [checkpoint_callback, history_logger], verbose = 1)
print('Training finished, plotting...')

plot_metrics(history)

