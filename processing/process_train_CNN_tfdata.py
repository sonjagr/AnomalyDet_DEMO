import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
import cv2
from common import *
from helpers.dataset_helpers import create_dataset
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
import random, time, argparse
import tensorflow as tf
from helpers.cnn_helpers import crop, patch_images,patch_labels, rotate_image, ds_length, grad, tf_bright_image

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
                    help="Which gpu to use", default="3", required=False)
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

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

## autoencode and return difference
@tf.function
def encode(img):
    img = tf.reshape(img, [-1, 2720, 3840, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.sqrt(tf.pow(tf.subtract(img, decoded_img), 2))
    return aed_img

random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

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
    anomalous_nbr_before = ds_length(anomalous_dataset)
    anomalous_images = anomalous_dataset.map(lambda x, y: x)
    anomalous_labels = anomalous_dataset.map(lambda x, y: y)
    if to_augment == True:
        rotated_dataset = anomalous_images.map(lambda x: rotate_image(x))
        rotated_anomalous_dataset = tf.data.Dataset.zip((rotated_dataset, anomalous_labels))
        combined_anomalous_dataset = anomalous_dataset.concatenate(rotated_anomalous_dataset)
        if to_brightness == True:
            bright_dataset = anomalous_images.map(tf_bright_image(x))
            bright_anomalous_dataset = tf.data.Dataset.zip((bright_dataset, anomalous_labels))
            bright_anomalous_dataset = bright_anomalous_dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
            combined_anomalous_dataset = combined_anomalous_dataset.concatenate(bright_anomalous_dataset)
    if to_augment == False:
        combined_anomalous_dataset = anomalous_dataset
    anomalous_nbr = ds_length(combined_anomalous_dataset)
    normal_dataset = dataset.filter(lambda x, y: y == 0.).shuffle(500).take(normal_times * anomalous_nbr)

    combined_dataset_batch = normal_dataset.concatenate(combined_anomalous_dataset).shuffle(20000, reshuffle_each_iteration = True).batch(batch_size=batch_size, drop_remainder=True)
    return combined_dataset_batch, anomalous_nbr_before, anomalous_nbr

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
X_train_det_list = X_train_det_list[:200]
Y_train_det_list = Y_train_det_list[:200]
X_test_det_list = X_test_det_list[:100]
Y_test_det_list = Y_test_det_list[:100]

N_det_test = int(len(X_test_det_list))
N_det_train = len(X_train_det_list)
print('Loaded number of train, test samples: ', N_det_train, N_det_test)
print('All loaded. Starting processing...')
time1 = time.time()

train_imgs = create_dataset(X_train_det_list)
train_lbls = tf.data.Dataset.from_tensor_slices(Y_train_det_list)

test_imgs = create_dataset(X_test_det_list)
test_lbls = tf.data.Dataset.from_tensor_slices(Y_test_det_list)

train_combined_dataset_batch, train_ano_before, train_ano = preprocess(train_imgs, train_lbls, to_encode = True, normal_times=50, to_augment= True, to_brightness=False, batch_size = batch_size)
val_combined_dataset_batch, val_ano_before, val_ano = preprocess(test_imgs, test_lbls, to_encode = True, normal_times=50, to_augment= False, batch_size = batch_size)
val_combined_dataset = val_combined_dataset_batch.unbatch()

train_dataset_len = ds_length(train_combined_dataset_batch.unbatch())
val_dataset_len = ds_length(val_combined_dataset)
print('Number of training, validation patches in total: ', train_dataset_len, val_dataset_len)
print('Number of anomalous training, validation patches before and after augment: ', train_ano_before, val_ano_before, train_ano, val_ano)

### PLOTTING FOR CROSSCHECK
w = 0
for x, y, in train_combined_dataset_batch.unbatch():
    if y == 1 and w < 10:
        plt.imshow(tf.reshape(x, [160,160]))
        plt.show()
        w = w+1
####

time2 = time.time()
pro_time = time2-time1
print('Processing (time {:.2f} s) finished, starting training...'.format(pro_time))

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

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
    model = model_tf

print(model.summary())

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

    for val_x, val_y in val_combined_dataset_batch:
        val_x = tf.reshape(val_x, [-1, 160, 160, 1])
        val_y = tf.cast(val_y, tf.float32)

        logits = model(val_x, training=False)
        epoch_val_loss_mean.update_state(bce(val_y, logits))
        prediction = tf.cast(logits, tf.uint64)
        epoch_val_accuracy.update_state(val_y, prediction)

    print("Test set accuracy: {:.3%}".format(epoch_val_accuracy.result()))

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
        plt.savefig('saved_CNNs/%s/loss_plot.png'% (savename))
        plt.show()
