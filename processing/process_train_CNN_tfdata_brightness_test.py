import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
from helpers.dataset_helpers import create_dataset
from old_codes.autoencoders import *
import matplotlib.pyplot as plt
import random, time, argparse
import tensorflow as tf
from helpers.cnn_helpers import ds_length, grad, preprocess

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

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

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

N_det_val = int(len(X_val_det_list))
## how many samples used
X_train_det_list = X_train_det_list
Y_train_det_list = Y_train_det_list
X_val_det_list = X_val_det_list[:int(N_det_val/2)]
Y_val_det_list = Y_val_det_list[:int(N_det_val/2)]

N_det_train = len(X_train_det_list)
print('Loaded number of train, val samples: ', N_det_train, N_det_val)
print('All loaded. Starting processing...')
time1 = time.time()

train_imgs = create_dataset(X_train_det_list)
train_lbls = tf.data.Dataset.from_tensor_slices(Y_train_det_list)

val_imgs = create_dataset(X_val_det_list)
val_lbls = tf.data.Dataset.from_tensor_slices(Y_val_det_list)

train_combined_dataset_batch, train_ano_before, train_ano = preprocess(train_imgs, train_lbls, to_encode = True, ae = ae, normal_times = 10, to_augment = True, to_brightness = False, to_brightness_before_ae=False, batch_size = batch_size,  drop_rem = True)
val_combined_dataset_batch, val_ano_before, val_ano = preprocess(val_imgs, val_lbls, to_encode = True, ae = ae, normal_times = 10, to_augment = False, batch_size = batch_size, drop_rem = False)
val_combined_dataset = val_combined_dataset_batch.unbatch()

train_dataset_len = ds_length(train_combined_dataset_batch.unbatch())
val_dataset_len = ds_length(val_combined_dataset)
print('Number of training, validation patches in total: ', train_dataset_len, val_dataset_len)
print('Number of anomalous training, validation patches before and after augment: ', train_ano_before, val_ano_before, train_ano, val_ano)

time2 = time.time()
pro_time = time2-time1
print('Processing (time {:.2f} s) finished, starting training...'.format(pro_time))

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

if load == 'True':
    print('Loading model...')
    model = tf.keras.models.load_model('saved_CNNs/%s/cnn_%s_epoch_%i' % (savename, savename,cont_epoch))
    cont_epoch = cont_epoch
    train_loss_results = np.load('saved_CNNs/%s/cnn_%s_train_loss.npy' % (savename, savename))
    val_loss_results = np.load('saved_CNNs/%s/cnn_%s_val_loss.npy' % (savename, savename))
else:
    cont_epoch = 0
    train_loss_results, val_loss_results = [], []
    train_accuracy_results, val_accuracy_results = [], []
    from CNNs import *
    if model_ID == 'model_tf':
        model = model_tf
    if model_ID == 'model_simple':
        model = model_simple

print(model.summary())

for epoch in tqdm(range(cont_epoch, num_epochs), total = num_epochs):
    epoch_train_loss_mean, epoch_val_loss_mean = [], []
    epoch_train_accuracy, epoch_val_accuracy =tf.keras.metrics.BinaryAccuracy, tf.keras.metrics.BinaryAccuracy

    for x, y in train_combined_dataset_batch:
        x = tf.reshape(x, [-1, 160, 160, 1])
        y = tf.cast(y, tf.float32)

        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_train_loss_mean = np.append(epoch_train_loss_mean, loss_value)
        y_hat =  model(x, training=True)
    train_loss_results.append(np.mean(epoch_train_loss_mean))

    for val_x, val_y in val_combined_dataset_batch:
        val_x = tf.reshape(val_x, [-1, 160, 160, 1])
        val_y = tf.cast(val_y, tf.float32)

        logits = model(val_x, training=False)
        epoch_val_loss_mean = np.append(epoch_val_loss_mean, bce(val_y, logits))
        y_hat = tf.cast(logits, tf.uint64)

    val_loss_results.append(np.mean(epoch_val_loss_mean))

    if epoch % 2 == 0:
        print("Epoch {:03d}: Train loss: {:.10f} Validation loss: {:.10f}".format(epoch, np.mean(epoch_train_loss_mean), np.mean(epoch_val_loss_mean)))
        try:
            os.makedirs('saved_cNNs/%s' % savename)
        except FileExistsError:
            pass
        model_saveto = 'saved_CNNs/%s/cnn_%s_epoch_%i' % (savename, savename, epoch)
        model.save(model_saveto)
        print('Model checkpoint saved to ', model_saveto)
        np.save('saved_CNNs/%s/cnn_%s_val_loss.npy' % (savename, savename), val_loss_results)
        np.save('saved_CNNs/%s/cnn_%s_train_loss.npy' % (savename, savename), train_loss_results)
        print('Model val and train losses saved to ', 'saved_CNNs/%s/' % savename)

        plt.plot(np.arange(0, len(val_loss_results)), val_loss_results, label = 'Val loss')
        plt.plot(np.arange(0, len(train_loss_results)), train_loss_results, label ='Train loss')
        plt.legend()
        plt.savefig('saved_CNNs/%s/loss_plot.png'% (savename))
        plt.show()
