import numpy as np
import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from helpers.dataset_helpers import create_dataset
from old_codes.autoencoders import *
from common import *

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_ae = "/"
dir_det = "DET/"

X_test_list = np.load(base_dir + dir_ae + 'X_test_AE.npy', allow_pickle=True)
X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)
np.random.seed(42)
X_test_val_list = np.random.choice(X_test_list, 2000, replace=False)
X_test_list, X_val_list = train_test_split(X_test_val_list, test_size=0.5, random_state=42)

#X_test_list_norm = [imgDir_gpu + s for s in X_val_list][:50]
X_test_list_det = [imgDir_gpu + s for s in X_test_det_list][:50]

#X_val_list_comb = X_test_list_norm + X_test_list_det
X_val_list_comb =  X_test_list_det

batch_size = 1

def resize(item):
    model = tf.keras.Sequential([tf.keras.layers.Cropping2D(cropping=((0, 16),(0, 0)))])
    return model(item)

val_dataset = create_dataset(X_val_list_comb, _shuffle=True).batch(batch_size)

xshape = 2720
val_dataset = val_dataset.map(lambda item: tuple(tf.py_function(resize, [item], [tf.float32,])))

model = 'TQ3'
savemodel = 'TQ3_more_data'
epoch = '500'
print('model: ', model , 'savemodel: ', savemodel, 'epoch: ', epoch)

saveloc = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/'+savemodel
Path(saveloc).mkdir(parents=True, exist_ok=True)

ae = AutoEncoder()
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_TQ3_more_data/AE_TQ3_318_to_318_epochs')

import pickle

with open('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_TQ3_more_data/cost.pkl', 'rb') as f:
    x = pickle.load(f)

test_losses = x['test_losses']
train_losses = x['train_losses']
print(len(train_losses))
#plt.plot(np.arange(1, len(test_losses) + 1, 1/16000), train_losses, c='y', label = 'Train', linestyle='--')
plt.plot(np.arange(1, len(test_losses) + 1, 1), test_losses, color='C1', label='Validation')
plt.ylabel('L2 loss', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.title('Validation loss during training', fontsize = 16)
plt.grid(zorder=-3)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

print(ae.encoder.summary())
print(ae.decoder.summary())

'''
with open('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/'+savemodel+'/cost.pkl', 'rb') as f:
    x = pickle.load(f)
test_losses = x['test_losses']
min_steps_test = x['min_steps_test']
plt.plot(np.arange(1,len(test_losses)+1,1), test_losses)
plt.ylabel('L loss (L2)')
plt.xlabel('Epoch')
plt.grid(zorder = -3)
plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/'+savemodel+'/test_loss.png')
plt.show()
'''

import matplotlib.pyplot as plt
def plot_image(image, i):
    image = image[0]
    fig, ax = plt.subplots()
    ax.imshow(image, vmin=0, vmax=255)
    #plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/'+savemodel+'/aed_img_%s.png' % i)
    plt.show()
    return image

def plot_orig_image(image, i):
    image = np.load(image)
    fig, ax = plt.subplots()
    ax.imshow(image, vmin=0, vmax=255)
    #plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/'+savemodel+'/orig_img_%s.png' % i)
    plt.show()
    return image[:xshape, :]

def plot_diff(image, i):
    fig, ax = plt.subplots()
    ax.imshow(image, vmin=0, vmax=255)
    #plt.savefig('/afs/cern.ch/user/s/sgroenro/anomaly_detection/plots/'+savemodel+'/diff_2_img_%s.png' % i)
    plt.show()

'''
i = 0
for x in tqdm(val_dataset.take(5), total=10):
    plt.figure(1)
    plt.imshow(x[0].numpy().reshape(2720, 3840), cmap = 'gray')
    plt.tick_params(axis='both',which='both',  bottom=False, right=False, left = False , labelbottom=False, labelleft = False , labelright =  False)
    plt.show()
    val_enc = ae.encode(x)
    dec = ae.decode(val_enc)
    plt.figure(2)
    plt.imshow(dec.numpy().reshape(2720, 3840), cmap = 'gray')
    plt.tick_params(axis='both', which='both', bottom=False, right=False, left = False , labelbottom=False, labelleft = False , labelright =  False)
    plt.show()
    diff = np.sqrt((x-dec) ** 2)
    plt.figure(3)
    plt.imshow(diff.reshape(2720, 3840), cmap = 'rainbow')
    plt.tick_params(axis='both', which='both', bottom=False, right=False, left = False , labelbottom=False, labelleft = False , labelright =  False)
    plt.show()
    plt.figure(4)
    plt.imshow(diff.reshape(2720, 3840), cmap='rainbow')
    xs = np.arange(1,25)
    ys = np.arange(1,18)
    for xline in xs:
        plt.plot(np.full(3, xline*160), [0, 1500, 2740],  linestyle = '-', color = 'white')
    for yline in ys:
        plt.plot( [0, 2000, 3860], np.full(3, yline*160), linestyle = '-', color = 'white')
    plt.tick_params(axis='both', which='both', bottom=False, right=False, left=False, labelbottom=False, labelleft=False, labelright=False)
    plt.show()
'''

def mean_error(X_val_list_comb, brightness):
    i=0
    recon_error = []
    for x in tqdm(X_val_list_comb, total=50):
        orig_img = x[0].numpy()
        orig_img = (orig_img*brightness).reshape(-1, xshape, 3840, 1)
        val_enc = ae.encode(orig_img)
        val_dec = ae.decode(val_enc)
        dec_img = val_dec
        diff = np.sqrt((dec_img - orig_img)**2)
        recon_error = np.append(recon_error, diff)
        diff = (dec_img - orig_img)
        #plot_diff(tf.reshape(diff, [2720, 3840]), i)
        i = i + 1
    return np.mean(recon_error)

def plot_mean_error(X_val_list_comb):
    mean_error1 = mean_error(X_val_list_comb, 1.)
    brightnesses = np.arange(0.5, 1.6, 0.1)
    errors_brightnesses = []
    mean_errors = []

    for i in brightnesses:
        errors_brightnesses.append(mean_error(X_val_list_comb, i))
        mean_errors.append(mean_error1*i)

    plt.plot(brightnesses, errors_brightnesses)
    plt.grid()
    plt.show()

    plt.plot(mean_errors, errors_brightnesses)
    plt.grid()
    plt.show()

plot_mean_error(val_dataset)