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
                    help="Load old model or not", default = "False", required=False)
parser.add_argument("--contfromepoch", type=int,
                    help="Epoch to continue training from", default=1, required=False)

args = parser.parse_args()

num_epochs = args.epochs
batch_size = args.batch_size
model_ID = args.model_ID
gpu = args.gpu
savename = args.savename
load = args.load
lr = args.lr
cont_epoch = args.contfromepoch

if gpu != 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

try:
    os.makedirs('saved_CNNs/%s' % savename)
except FileExistsError:
    pass

f = open('saved_CNNs/%s/argfile.txt' % savename, "w")
f.write("Epochs: %s" % num_epochs)
f.write("Batch_size: %s" % batch_size)
f.write("Model_ID: %s" % model_ID)
f.write("Savename: %s" % savename)
f.write("Lr: %s" % lr)
f.close()

random.seed(42)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

ae = AutoEncoder()
print('Loading autoencoder and data...')
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_500_to_500_epochs')

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_val_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)
X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_val_det_list = [images_dir_loc + s for s in X_val_det_list]
Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET.npy', allow_pickle=True).tolist()
Y_val_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()

N_det_val = int(len(X_val_det_list)/2)
X_train_det_list = X_train_det_list
Y_train_det_list = Y_train_det_list
X_val_det_list = X_val_det_list[:int(N_det_val/2)]
Y_val_det_list = Y_val_det_list[:int(N_det_val/2)]

N_det_train = len(X_train_det_list)
print('Loaded number of train, val samples: ', N_det_train, N_det_val)
print('All loaded. Starting processing...')
time1 = time.time()

train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=True)

val_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list, _shuffle=True)

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
        plt.imshow(x)
        plt.title(str(y))
        plt.show()

@tf.function
def crop(img, lbl):
    img = tf.reshape(img, [-1, 2736, 3840, INPUT_DIM])
    img = tf.keras.layers.Cropping2D(cropping=((0, 16), (0, 0)))(img)
    return img, lbl

@tf.function
def encode(img, lbl, ae):
    img = tf.reshape(img, [-1, 2720, 3840, INPUT_DIM])
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
def process_crop_encode(image, label):
    image, label = crop(image, label)
    image, label = encode(image, label, ae)
    return image,label

@tf.function
def rotate(image, label):
    image = tf.reshape(image, [1, 160, 160])
    rots = tf.random.uniform([], minval=1, maxval=3, dtype=tf.int32)
    rot = tf.image.rot90(image, k=rots)
    return tf.reshape(rot, [-1]), label

## change brightness of patch
def change_bright(image, label):
    delta = random.choice([-0.2, 0.2, 0.01])
    image = tf.image.adjust_brightness(image, delta)
    return image, label

@tf.function
def augment(image_label, seed):
    image, label = image_label
    if label == 1.:
        image, label = rotate(image, label, seed)
    return image, label

@tf.function
def format(image, label):
    image = tf.reshape(image, [160, 160, 1])
    label = tf.cast(label, tf.float32)
    return image, label

counter = tf.data.experimental.Counter()
train_ds = train_ds.map(process_crop_encode)
val_ds = val_ds.map(process_crop_encode)

train_ds = train_ds.flat_map(patch_images).unbatch()
val_ds = val_ds.flat_map(patch_images).unbatch()

train_ds_rotated = train_ds.filter(lambda x, y: y == 1.).map(rotate)

train_ds_rotated_bright = train_ds_rotated.map(change_bright)
train_ds_bright = train_ds.filter(lambda x, y: y == 1.).map(change_bright)

#train_ds = train_ds.concatenate(train_ds_rotated)
train_ds_with_rot = tf.data.experimental.sample_from_datasets([train_ds,train_ds_rotated], seed=42)
#train_ds = train_ds.concatenate(train_ds_rotated_bright)
train_ds_with_rot_bright = tf.data.experimental.sample_from_datasets([train_ds_with_rot,train_ds_rotated_bright], seed=42)
#train_ds = train_ds.concatenate(train_ds_bright)
train_ds_final = tf.data.experimental.sample_from_datasets([train_ds_with_rot_bright,train_ds_bright], seed=42)

train_ds_final = train_ds_final.map(format)
val_ds = val_ds.map(format)

train_ds_anomaly = train_ds_final.filter(lambda x, y: y == 1.)
train_ds_normal = train_ds_final.filter(lambda x, y: y == 0.)

#normal, anomalous = len(list(train_ds_normal)), len(list(train_ds_anomaly))
#anomaly_weight = normal/anomalous

#print('Number of normal, anomalous samples: ', normal, anomalous)
#print('Anomaly weight: ', anomaly_weight)

anomaly_weight = 50.

train_ds_batch = train_ds_final.batch(batch_size=batch_size, drop_remainder = True)
val_ds_batch = val_ds.batch(batch_size=batch_size)

#plot_examples(train_ds_anomaly.shuffle(100).take(20))
#plot_examples(train_ds_normal.shuffle(100).take(20))
#anomaly = len(list(train_ds_anomaly))
#normal = len(list(train_ds_normal))
#print('Number of anomalous, normal training patches: ', anomaly, normal)

time2 = time.time()
pro_time = time2-time1
print('Processing (time {:.2f} s) finished, starting training...'.format(pro_time))

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

if load == 'True':
    print('Loading model...')
    model = tf.keras.models.load_model('/afs/cern.ch/user/s/sgroenro/anomaly_detection/saved_CNNs/%s/cnn_%s' % (savename, savename))
else:
    from CNNs import *
    if model_ID == 'model_tf':
        model = model_tf
    if model_ID == 'model_tf2':
        model = model_tf2
    if model_ID == 'model_tf3':
        model = model_tf3
    if model_ID == 'model_simple':
        model = model_simple

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'),
]


print(model.summary())
model.compile(optimizer = optimizer, loss= tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=METRICS)

class_weights = {0: 1., 1: anomaly_weight}

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='saved_CNNs/%s/cnn_%s' % (savename, savename),
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=False)

filename = 'saved_CNNs/%s/history_log.csv' % savename
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=load)

def scheduler(epoch, lr):
  if epoch < 50:
    return lr
  else:
    return lr * tf.math.exp(-0.01)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

print('Starting training:')
history = model.fit(train_ds_batch, epochs = num_epochs, validation_data = val_ds_batch, class_weight = class_weights, callbacks = [checkpoint_callback, history_logger, lr_schedule])
print('Training finished, plotting...')

plot_metrics(history)


