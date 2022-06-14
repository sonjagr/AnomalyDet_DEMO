import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, log_loss
from old_codes.autoencoders import *
import random, argparse
import matplotlib.pyplot as plt
random.seed(42)

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--model_ID", type=str, help="Model ID",
                    default="model", required=True)
parser.add_argument("--batch_size", type=int,
                    help="Batch size for the training", default=64, required=False)
parser.add_argument("--epochs", type=int,
                    help="Number of epochs for the training", default=50, required=False)
parser.add_argument("--lr", type=float,
                    help="Learning rate", default=1e-3, required=False)
parser.add_argument("--is_ae", type=int,
                    help="Are images aed", default=0, required=True)
parser.add_argument("--gpu", type=str,
                    help="Which gpu to use", default="3", required=False)
parser.add_argument("--savename", type=str,
                    help="Name to save with", required=True)
parser.add_argument("--load", type=str,
                    help="Load old model or not", default = "False", required=False)
parser.add_argument("--contfromepoch", type=int,
                    help="Epoch to continue training from", default=1, required=False)
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
model_ID = args.model_ID
gpu = args.gpu
savename = args.savename
load = args.load
lr = args.lr
is_ae = args.is_ae
cont_epoch = args.contfromepoch

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

try:
    os.makedirs('saved_CNNs/%s' % savename)
except FileExistsError:
    pass

f = open('saved_CNNs/%s/argfile.txt' % savename, "w")
f.write("   Epochs: %s" % epochs)
f.write("   Batch_size: %s" % batch_size)
f.write("   Model_ID: %s" % model_ID)
f.write("   Savename: %s" % savename)
f.write("   Lr: %s" % lr)

f.close()

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

if load == 'True':
    print('Loading model...')
    model = tf.keras.models.load_model('saved_CNNs/%s/cnn_%s_epoch_%i' % (savename, savename,cont_epoch))
    val_scores = np.load('saved_CNNs/%s/cnn_%s_test_loss.npy' % (savename, savename))
    val_scores = list(val_scores)
    training_scores = np.load('saved_CNNs/%s/cnn_%s_train_loss.npy' % (savename, savename))
    training_scores = list(training_scores)

else:
    cont_epoch = 1
    training_scores, val_scores = list(), list()
    from old_codes.CNNs import *
    if model_ID == 'model_tf':
        model = model_tf
    if model_ID == 'model_tf2':
        model = model_tf2
    if model_ID == 'model_simple':
        model = model_simple
    if model_ID == 'model_simple2':
        model = model_simple2

print(model.summary())

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,decay_steps=10000,decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['binary_crossentropy'])

# which dataset to use
name = '19'

train_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'train_img_list_aug_%s.npy' % name)
train_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'train_lbl_list_aug_%s.npy' % name)

val_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/' + 'val_img_list_%s.npy' % name)
val_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/' + 'val_lbl_list_%s.npy' % name)

print('max of training data', np.max(train_img_list))
print('max of validation data', np.max(val_img_list))

print(np.min(train_img_list))
print(np.min(val_img_list))

from sklearn.utils import shuffle

print('Train shape', len(train_img_list), len(train_lbl_list))
print('Validation shape', len(val_img_list), len(val_lbl_list))

batches = np.floor(len(train_img_list)/batch_size)
cut = int(np.floor(np.mod(len(train_img_list), batches)))

class_weights = {0: 1., 1: 20.}

for epoch in range(cont_epoch, epochs):
    print("\nStart of epoch %d" % (epoch,))
    train_scores_epoch = []
    b = 0
    train_img_list, train_lbl_list = shuffle(train_img_list, train_lbl_list)
    train_img_list_cut = train_img_list[:-cut]
    train_lbl_list_cut = train_lbl_list[:-cut]

    train_img_batches = np.split(train_img_list_cut, batches, axis=0)
    train_lbl_batches = np.split(train_lbl_list_cut, batches, axis=0)
    print(np.array(train_img_batches).shape)
    for x_batch, y_batch in tqdm(zip(train_img_batches, train_lbl_batches), total=len(train_lbl_batches)):
        model.fit(x_batch, y_batch, class_weight = class_weights, verbose = 0)
        train_loss_batch = log_loss(y_batch, model.predict(x_batch).astype("float64"), labels = [0,1])
        train_scores_epoch.append(train_loss_batch)
        training_scores.append(train_loss_batch)
    print('Training loss: ', np.mean(train_scores_epoch))

    model_saveto = 'saved_CNNs/%s/cnn_%s_epoch_%i' % (savename, savename, epoch)
    model.save(model_saveto)
    print('Model checkpoint saved to ', model_saveto)

    print('\nStarting testing:')

    val_pred = model.predict(val_img_list)
    val_loss = log_loss(val_lbl_list, val_pred.astype("float64"))
    val_scores.append(val_loss)
    print('Validation score: ', val_loss)
    print(val_lbl_list[:6])
    print(val_pred[:6].flatten())

    tn, fp, fn, tp = confusion_matrix(val_lbl_list, np.round(val_pred)).ravel()
    print('Validation tn, fp, fn, tp:  ', tn, fp, fn, tp)

    np.save('saved_CNNs/%s/cnn_%s_test_loss.npy' % (savename, savename), val_scores)
    np.save('saved_CNNs/%s/cnn_%s_train_loss.npy' % (savename, savename), training_scores)
    print('Model validation and train losses saved to ', 'saved_CNNs/%s/' % savename)

    plt.plot(np.arange(0, len(val_scores), (len(val_scores) / len(training_scores))), training_scores, label='Train loss')
    plt.plot(np.arange(0, len(val_scores), 1), val_scores, label='Validation loss')
    plt.legend()
    plt.grid()
    plt.title(str(savename))
    plt.savefig('saved_CNNs/%s/loss_plot.png' % (savename))
    plt.show()