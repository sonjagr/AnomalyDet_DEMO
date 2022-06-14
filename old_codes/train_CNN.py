import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from old_codes.autoencoders import *
import random, argparse
random.seed(42)

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

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
@tf.function
def compute_loss(model, x, y_ref):
    y = model(x)
    reconstruction_error = bce(y_ref, y)
    return reconstruction_error

@tf.function
def train_step(model, x, y_ref, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y_ref)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

if load == 'True':
    print('Loading model...')
    model = tf.keras.models.load_model('saved_class/%s/cnn_%s_epoch_%i' % (savename, savename,cont_epoch))
    testing_scores = np.load('losses/%s/test_loss_%s.npy' % (savename, savename))
    testing_scores = list(testing_scores)
    training_scores = np.load('losses/%s/train_loss_%s.npy' % (savename, savename))
    training_scores = list(training_scores)

else:
    cont_epoch = 1
    training_scores, testing_scores = list(), list()
    from old_codes.CNNs import *
    if model_ID == 'model_works_newdatasplit4':
        model = model_works_newdatasplit4
    if model_ID == 'model_maxpool':
        model = model_maxpool

print(model.summary())

optimizer = tf.keras.optimizers.Adam(lr)

# which dataset to use
name = '4'

train_img_list = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'train_img_list_%s.npy' % name)
train_lbl_list = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'train_lbl_list_%s.npy' % name)

test_img_list = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'val_img_list_%s.npy' % name)
test_lbl_list = np.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/processed/'+'val_lbl_list_%s.npy' % name)

processed_train_dataset = tf.data.Dataset.from_tensor_slices((train_img_list, train_lbl_list)).shuffle(train_img_list.shape[0], reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
processed_test_dataset = tf.data.Dataset.from_tensor_slices((test_img_list, test_lbl_list)).batch(1)

for epoch in range(cont_epoch, epochs):
    print("\nStart of epoch %d" % (epoch,))
    train_scores_epoch = []
    b = 0
    for x_batch, y_batch in tqdm(processed_train_dataset, total=tf.data.experimental.cardinality(processed_train_dataset).numpy()):
        train_step(model, x_batch, y_batch, optimizer)
        loss = tf.keras.metrics.Mean()
        loss(compute_loss(model, x_batch, y_batch))
        mean_cross_entr = loss.result()
        training_scores.append(mean_cross_entr.numpy())

    print('Training loss: ', mean_cross_entr.numpy())
    try:
        os.makedirs('saved_class/%s' % savename)
    except FileExistsError:
        pass
    model_saveto = 'saved_class/%s/cnn_%s_epoch_%i' % (savename, savename, epoch)
    model.save(model_saveto)
    print('Model checkpoint saved to ', model_saveto)

    print('\nStarting testing:')

    loss = tf.keras.metrics.Mean()
    for test_x, y_ref in processed_test_dataset:
        loss(compute_loss(model, test_x, y_ref))
    test_loss = loss.result().numpy()
    testing_scores.append(test_loss)

    test_pred = model(test_img_list)
    print('Testing score: ', test_loss)

    tn, fp, fn, tp = confusion_matrix(test_lbl_list, np.round(test_pred)).ravel()
    print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
    try:
        os.makedirs('losses/%s' % savename)
    except FileExistsError:
        pass

    np.save('losses/%s/test_loss_%s.npy' % (savename, savename), testing_scores)
    np.save('losses/%s/train_loss_%s.npy' % (savename, savename), training_scores)
    print('Model test and train losses saved to ','losses/%s/' % savename)