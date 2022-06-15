import numpy as np
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
    ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_TQ3_more_data/AE_TQ3_318_to_318_epochs')


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

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
dir_ae = 'AE/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

## extract normal images for training
X_train_list_ae = np.load(base_dir + dir_ae + 'X_train_AE.npy', allow_pickle=True)
X_test_list_ae = np.load(base_dir + dir_ae + 'X_test_AE.npy', allow_pickle=True)

np.random.seed(1)
X_train_list_normal_to_remove = np.random.choice(X_train_list_ae, 1600, replace=False)
X_test_val_list_normal_to_remove = np.random.choice(X_test_list_ae, 4000, replace=False)

X_train_list_normal_removed = [x for x in X_train_list_ae if x not in X_train_list_normal_to_remove]
X_test_list_normal_removed = [x for x in X_test_list_ae if x not in X_test_val_list_normal_to_remove]

## load the images containing anomalies
X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET_very_cleaned.npy', allow_pickle=True)
X_val_det_list = np.load(base_dir + dir_det + 'X_test_DET_very_cleaned.npy', allow_pickle=True)
X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_val_det_list = [images_dir_loc + s for s in X_val_det_list]

Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET_very_cleaned.npy', allow_pickle=True).tolist()
Y_val_det_list = np.load(base_dir + dir_det + 'Y_test_DET_very_cleaned.npy', allow_pickle=True).tolist()

## split test and validation sets
N_det_val = int(len(X_val_det_list)/2)
N_det_train = len(X_train_det_list)
X_val_det_list = X_val_det_list[:N_det_val]
Y_val_det_list = Y_val_det_list[:N_det_val]

np.random.seed(42)
X_train_normal_list = np.random.choice(X_train_list_normal_removed, int(N_det_train), replace=False)
X_val_normal_list = np.random.choice(X_test_list_normal_removed, int(N_det_val), replace=False)
X_train_normal_list = [images_dir_loc + s for s in X_train_normal_list]
X_val_normal_list = [images_dir_loc + s for s in X_val_normal_list]

print('    Loaded number of train, val samples: ', N_det_train, N_det_val)
print('All loaded. Starting dataset creation...')
time1 = time.time()

## combine anomalous and normal images
X_train_list = np.append(X_train_det_list, X_train_normal_list)
X_val_list = np.append(X_val_det_list, X_val_normal_list)

Y_train_list = np.append(Y_train_det_list, np.full((int(len(X_train_normal_list)), 408), 0.), axis = 0)
Y_val_list = np.append(Y_val_det_list, np.full((int(len(X_val_normal_list)), 408), 0.), axis = 0)

## only with defects
train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=False)
val_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list, _shuffle=False)

normal_normal_train_ds = create_cnn_dataset(X_train_normal_list, np.full((int(len(X_train_normal_list)), 408), 0.), _shuffle=False)

## with normal images
more_normal_train_ds = create_cnn_dataset(X_train_list, Y_train_list, _shuffle=False)
more_normal_val_ds = create_cnn_dataset(X_val_list, Y_val_list, _shuffle=False)

#train_ds = more_normal_train_ds
#val_ds = more_normal_val_ds

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

if load == 'True':
    print('    Loading previously trained model...')
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
    if model_ID == 'model_whole':
        model = model_whole
    if model_ID == 'model_whole_smaller':
        model = model_whole_smaller
    if model_ID == 'model_whole_larger':
        model = model_whole_larger
    def scheduler(epoch, lr):
        if epoch < 50:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

@tf.function
def encode(img, lbl, ae):
    img = tf.reshape(img, [-1, 2720, 3840, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.abs(tf.subtract(img, decoded_img))
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
    aed_img = tf.abs(tf.subtract(img, decoded_img))
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
    image = tf.reshape(image, [160, 160, 1])
    rot = tf.image.rot90(image, k=rots)
    return tf.reshape(rot, [160,160,1]), label

@tf.function
def flip(image_label, seed):
    image, label = image_label
    image = tf.reshape(image, [160, 160, 1])
    if seed > 5:
        flipped = tf.image.flip_left_right(image)
    else:
        flipped = tf.image.flip_up_down(image)
    return tf.reshape(flipped, [160,160,1]), label

@tf.function
def shift(image_label, factor):
    image, label = image_label
    image = tf.reshape(image, [160, 160, 1])
    shifted = tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=factor, width_factor=factor, fill_mode='reflect', interpolation='bilinear')
    image = shifted
    return tf.reshape(shifted, [160,160,1]), label

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
    val_ds = val_ds.map(process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

if not use_ae:
    train_ds_nob = train_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

## add changes in brighness
np.random.seed(42)

if bright_aug:
    brightnesses = np.random.uniform(low = 0.7, high = 0.9, size = np.array(Y_train_det_list).shape[0])
    counter1 = tf.data.Dataset.from_tensor_slices(brightnesses)
    train_ds_c = tf.data.Dataset.zip((train_ds, counter1))
    train_ds_brightness = train_ds_c.map(lambda x,z: process_crop_bright_encode(x,z), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds_brightness = train_ds_brightness.flat_map(patch_images).unbatch()

## apply patching
train_ds = train_ds_nob.flat_map(patch_images).unbatch()
val_ds = val_ds.flat_map(patch_images).unbatch()

train_ds_anomaly = train_ds.filter(lambda x, y:  y == 1.)

if bright_aug:
    train_ds_brightness_anomaly = train_ds_brightness.filter(lambda x, y:  y == 1.)
    train_ds_anomaly = train_ds_anomaly.concatenate(train_ds_brightness_anomaly)

nbr_anom_patches = 1283
nbr_anom_patches = len(list(train_ds.filter(lambda x, y:  y == 1.)))
print('    Number of anomalous patches: ', nbr_anom_patches)

if bright_aug:
    aug_size = 2*nbr_anom_patches
if not bright_aug:
    aug_size = nbr_anom_patches

rotations = np.random.randint(low = 1, high = 4, size = aug_size).astype('int32')
counter2 = tf.data.Dataset.from_tensor_slices(rotations)
train_ds_to_rotate = tf.data.Dataset.zip((train_ds_anomaly, counter2))

flip_seeds = np.random.randint(low = 1, high = 11, size = aug_size).astype('int32')
counter3 = tf.data.Dataset.from_tensor_slices(flip_seeds)
train_ds_to_flip = tf.data.Dataset.zip((train_ds_anomaly, counter3))

train_ds_rotated = train_ds_anomaly.concatenate(train_ds_to_rotate.map(rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE))
train_ds_rotated_flipped = train_ds_rotated.concatenate(train_ds_to_flip.map(flip, num_parallel_calls=tf.data.experimental.AUTOTUNE))

## shift does not work because it might shift some anomalies out of image, same with zoom/crop...
#shift_factors = np.random.randint(low = 10, high = 20, size = aug_size).astype('float64')
#shift_factors = shift_factors/100.
#counter4 = tf.data.Dataset.from_tensor_slices(shift_factors)
#train_ds_to_shift = tf.data.Dataset.zip((train_ds_rotated_flipped, counter4))

#train_ds_rotated_flipped_shifted = train_ds_rotated_flipped.concatenate(train_ds_to_shift.map(shift, num_parallel_calls=tf.data.experimental.AUTOTUNE))

augmented = train_ds_rotated_flipped

train_ds_normal = train_ds.filter(lambda x,y: y == 0.)

train_ds_final = train_ds_normal.concatenate(augmented)
train_ds_final = train_ds_final.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds_anomaly = train_ds_final.filter(lambda x, y:  y == 1.)

normal, anomalous = len(Y_train_det_list)*408-nbr_anom_patches, 6*nbr_anom_patches
#normal, anomalous = len(list(train_ds_normal)), len(list(augmented))
#normal, anomalous = 452413, 5132
anomaly_weight = normal/anomalous
print('    Number of normal, anomalous samples: ', normal, anomalous)
print('    Anomaly weight: ', anomaly_weight)

train_ds_final = train_ds_final.cache().shuffle(buffer_size=normal+anomalous, reshuffle_each_iteration=True)
val_ds = val_ds.cache()

train_ds_batch = train_ds_final.batch(batch_size=batch_size, drop_remainder = True)
val_ds_batch = val_ds.batch(batch_size=batch_size, drop_remainder = False)

plot_examples(train_ds_anomaly.take(2))
plot_examples(train_ds_normal.take(2))

time2 = time.time()
pro_time = time2-time1
print('Dataset (time {:.2f} s) created, starting training...'.format(pro_time))

optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)

print(model.summary())
model.compile(optimizer = optimizer, loss = tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=METRICS)

class_weights = {0: 1., 1: anomaly_weight}

filepath = 'saved_CNNs/%s/cnn_%s_epoch_{epoch:02d}' % (savename, savename)

##save every 10th epoch
save_every = int(np.floor((normal+anomalous)/batch_size)*10)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq=save_every)

##save training history to file
filename = 'saved_CNNs/%s/history_log.csv' % savename
history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=load)

##add lr schedule
lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

print('Starting training:')

def dyn_weighted_bincrossentropy(true, pred):
    num_pred = tf.keras.backend.sum(tf.keras.backend.cast(pred < 0.5, true.dtype)) + tf.keras.backend.sum(true)
    zero_weight = tf.keras.backend.sum(true) / num_pred + tf.keras.backend.epsilon()
    one_weight = tf.keras.backend.sum(tf.keras.backend.cast(pred < 0.5, true.dtype)) / num_pred + tf.keras.backend.epsilon()
    weights = (1.0 - true) * zero_weight + true * one_weight
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bin_crossentropy =bce(true, pred)

    weighted_bin_crossentropy = weights * bin_crossentropy
    return tf.keras.backend.mean(weighted_bin_crossentropy)

def loss(model, x, y, training):
  y_ = model(x, training=training)
  return dyn_weighted_bincrossentropy(true=y, pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

train_loss_results = []
train_pres_results = []
train_recall_results = []
train_auc_results = []

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean(),
    epoch_pres = tf.keras.metrics.Precision(name='precision'),
    epoch_recall = tf.keras.metrics.Recall(name='recall'),
    epoch_auc = tf.keras.metrics.AUC(name='auc'),

    for x, y in train_ds_batch.prefetch(tf.data.experimental.AUTOTUNE):
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        epoch_pres.update_state(y, model(x, training=True))
        epoch_recall.update_state(y, model(x, training=True))
        epoch_auc.update_state(y, model(x, training=True))

    train_loss_results.append(epoch_loss_avg.result())
    train_pres_results.append(epoch_pres.result())
    train_recall_results.append(epoch_recall.result())
    train_auc_results.append(epoch_auc.result())

    print("TRAINING Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Precision: {:.3%}, AUC: {:.3%} ".format(epoch, epoch_loss_avg.result(), epoch_pres.result(), epoch_recall.result(), epoch_auc.result()))

    test_loss = loss(training=False)
    test_pres = tf.keras.metrics.Precision()
    test_recall = tf.keras.metrics.Recall()

    for (x, y) in val_ds_batch.prefetch(tf.data.experimental.AUTOTUNE):
        logits = model(x, training=False)
        prediction = tf.math.argmax(logits, axis=1, output_type=tf.int64)
        test_loss = test_loss(y, prediction)
        test_pres(prediction, y)
        test_recall(prediction, y)

    print("Test set loss: {:.3%}, precision: {:.3%}, recall: {:.3%}".format(test_loss.result(), test_pres.result(), test_recall.result()))

print('Training finished')


