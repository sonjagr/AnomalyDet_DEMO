import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, log_loss
from helpers.dataset_helpers import create_cnn_dataset
from autoencoders import *
from common import *
from scipy.ndimage.interpolation import rotate
import random, argparse
import matplotlib.pyplot as plt
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

def classifier_scores(test_loss, train_loss, title):
    plt.plot(np.arange(0,len(test_loss)), test_loss, label = 'Test loss')
    plt.plot(np.arange(0,len(train_loss)), train_loss, linestyle= '--', label = 'Train loss')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Binary cross-entropy')
    plt.title(title)
    plt.show()

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
    from CNNs import *
    if model_ID == 'model_init2_lessr2':
        model = model_init2_lessr2
    if model_ID == 'model_init2_lessr2_larger':
        model = model_init2_lessr2_larger
    if model_ID == 'model_init2_lessr2_larger2':
        model = model_init2_lessr2_larger2
    if model_ID == 'model_init2_lessr2_larger_noae':
        model = model_init2_lessr2_larger_noae
    if model_ID == 'model_init2_lessr2_larger3':
        model = model_init2_lessr2_larger3
    if model_ID == 'model_works3':
        model = model_works3
    if model_ID == 'model_works_newdatasplit':
        model = model_works_newdatasplit
    if model_ID == 'model_works_simplest':
        model = model_works_simplest

print(model.summary())

optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['binary_crossentropy'])

#how many normal images for each defective
MTN = 1
if is_ae == 1:
    print('No AE')
    train_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'train_img_list_noae_%i.npy' % MTN)
    train_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'train_lbl_list_noae_%i.npy' % MTN)
    test_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'val_img_list_noae_%i.npy' % MTN)
    test_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'val_lbl_list_noae_%i.npy' % MTN)
if is_ae == 0:
    print('AE is used')
    train_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'train_img_list_%i.npy' % MTN)
    train_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'train_lbl_list_%i.npy' % MTN)
    test_img_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'val_img_list_%i.npy' % MTN)
    test_lbl_list = np.load('/data/HGC_Si_scratch_detection_data/processed/'+'val_lbl_list_%i.npy' % MTN)

def del_negs(lista):
    for i in range(0, len(lista)):
        minim = np.min(lista[i])
        if minim < 0:
            lista[i] = lista[i]+abs(minim)
    return lista


print(np.min(train_img_list))
print(np.min(test_img_list))

from sklearn.utils import shuffle

for epoch in range(cont_epoch, epochs):
    print("\nStart of epoch %d" % (epoch,))
    train_scores_epoch = []
    b = 0
    train_img_list, train_lbl_list = shuffle(train_img_list, train_lbl_list)
    train_img_list_cut = train_img_list[:-12]
    train_lbl_list_cut = train_lbl_list[:-12]

    train_img_batches =  np.split(train_img_list_cut, 50, axis=0)
    train_lbl_batches = np.split(train_lbl_list_cut, 50, axis=0)
    print(np.array(train_img_batches).shape)
    for x_batch, y_batch in tqdm(zip(train_img_batches, train_lbl_batches), total=50):
        model.fit(x_batch, y_batch, verbose = 0)
        train_scores_epoch.append(log_loss(y_batch, model.predict(x_batch).astype("float64")))

    training_scores.append(np.mean(train_scores_epoch))
    print('Training loss: ', np.mean(train_scores_epoch))
    try:
        os.makedirs('saved_class/%s' % savename)
    except FileExistsError:
        pass
    model_saveto = 'saved_class/%s/cnn_%s_epoch_%i' % (savename, savename, epoch)
    model.save(model_saveto)
    print('Model checkpoint saved to ', model_saveto)

    print('\nStarting testing:')


    test_pred = model.predict(test_img_list)
    test_loss = log_loss(test_lbl_list, test_pred.astype("float64"))
    testing_scores.append(test_loss)
    print('Testing score: ', test_loss)
    print(test_lbl_list[:6])
    print(test_pred[:6])

    tn, fp, fn, tp = confusion_matrix(test_lbl_list, np.round(test_pred)).ravel()
    print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
    try:
        os.makedirs('losses/%s' % savename)
    except FileExistsError:
        pass
    np.save('losses/%s/test_loss_%s.npy' % (savename, savename), testing_scores)
    np.save('losses/%s/train_loss_%s.npy' % (savename, savename), training_scores)
    print('Model test and train losses saved to ','losses/%s/' % savename)