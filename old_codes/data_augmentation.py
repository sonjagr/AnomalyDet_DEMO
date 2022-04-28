import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from helpers.dataset_helpers import create_dataset, create_cnn_dataset, box_index_to_coords
from autoencoders import *
import cv2
from common import *
from scipy.ndimage.interpolation import rotate
import random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
random.seed(42)

tf.executing_eagerly()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print('testi')

def rotate_img(img):
    img = img.reshape(160,160)
    rot_angle = random.choice([90, 180, 270])
    rot = rotate(img, rot_angle)
    return rot.flatten()

def change_bright_img(img):
    img = img.flatten()
    change = np.random.randint(low=80, high=120, size=1)/100
    img = img*change
    return img.flatten()

def split_img(item):
    tf.image.extract_patches(images=item,sizes=[1, 160, 160, 1],strides=[1, 160, 160, 1],rates = [1,1,1,1], padding='VALID')

def encode_split(img, i_a):
    img = img[0].numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = np.sqrt((decoded_img - img)**2)
    split_img = tf.image.extract_patches(images=aed_img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1],rates=[1, 1, 1, 1], padding='VALID')
    return split_img.numpy().reshape(17 * 24, 160 * 160)

bce = tf.keras.losses.BinaryCrossentropy()
@tf.function
def compute_loss(model, x, y_ref):
    y = model(x)
    reconstruction_error = bce(y_ref, y)
    return reconstruction_error

# define the training step
@tf.function
def train_step(model, x, y_ref, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y_ref)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

base_dir = '../db/'
dir_det = 'DET/'
images_dir_loc = '/data/HGC_Si_scratch_detection_data/MeasurementCampaigns/'

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)

X_train_det_list = [images_dir_loc + s for s in X_train_det_list]
X_test_det_list = [images_dir_loc + s for s in X_test_det_list]

Y_train_det_list = np.load(base_dir + dir_det + 'Y_train_DET.npy', allow_pickle=True).tolist()
Y_test_det_list = np.load(base_dir + dir_det + 'Y_test_DET.npy', allow_pickle=True).tolist()

train_def_dataset = create_cnn_dataset(X_train_det_list, Y_train_det_list, _shuffle=False)
test_def_dataset = create_cnn_dataset(X_test_det_list, Y_test_det_list, _shuffle=False)

N_det_test = len(X_test_det_list)
N_det_train = len(X_train_det_list)

train_def_dataset = train_def_dataset.take(200)
test_def_dataset = test_def_dataset.take(40)

ae = AutoEncoder()
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_354_to_354_epochs')

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(4,4)  , padding="valid",activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(8,8), strides=(8,8) , padding="valid" ,activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(1,1) , padding="valid", activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(4, 4), activation='elu', padding="valid"),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

#model = tf.keras.models.load_model('saved_class/saved_cnn_3')
#print(model.summary())

optimizer = tf.keras.optimizers.Adam(0.001)

epochs = 100
MTN=10
batch_size =4

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    defects, test_defects = 0, 0
    train_def_dataset.shuffle(100)
    train_scores = []
    for X_train, Y_train in tqdm(train_def_dataset, total=tf.data.experimental.cardinality(train_def_dataset).numpy()):
        Y_train = Y_train.numpy().reshape(17 * 24)
        i_a = np.where(Y_train == 1)[0]
        i_n = np.where(Y_train == 0)[0]
        split_img = encode_split(X_train, i_a)
        defects = defects + len(i_a)
        if len(i_a) > 0:
            def_imgs = split_img[i_a, :]
            rotated = np.array([rotate_img(item) for item in def_imgs])
            rotated_imgs = np.append(def_imgs, rotated, axis = 0)
            bright_imgs = np.array([change_bright_img(item) for item in rotated_imgs])
            def_imgs = np.append(rotated_imgs, bright_imgs, axis = 0)
            n_normal = def_imgs.shape[0]
            #i_n = random.choices(i_n, k=n_normal*MTN)
            norm_imgs = split_img[i_n,:]
            train_lbls = np.append(np.full(len(def_imgs), 1), np.full(len(norm_imgs),0), axis = 0)
            train_imgs = np.append(def_imgs, norm_imgs, axis = 0).reshape(-1, 160, 160)
            train_imgs, train_lbls = shuffle(train_imgs[:360], train_lbls[:360])
            img_batches = np.split(train_imgs, int(len(train_imgs)/batch_size), axis = 0)
            lbl_batches = np.split(train_lbls, int(len(train_imgs)/batch_size), axis = 0)
            for i in range(0,len(img_batches)):
                train_step(model, img_batches[i].reshape(-1, 160, 160), lbl_batches[i], optimizer)
                train_scores = np.append(train_scores, compute_loss(model, img_batches[i].reshape(-1, 160, 160), lbl_batches[i]))
    print('\nTraining loss: ', np.mean(train_scores, axis = 0))
    model.save('saved_class/saved_cnn_3')
    print('Starting testing:')
    test_scores = []
    test_lbl_list, test_img_list = [], []
    for X_test, Y_test in tqdm(test_def_dataset, total=tf.data.experimental.cardinality(test_def_dataset).numpy()):
        Y_test = Y_test.numpy().reshape(17*24)
        i_a = np.where(Y_test == 1)[0]
        i_n = np.where(Y_test == 0)[0]
        split_img = encode_split(X_test, i_a)
        test_defects = test_defects + len(i_a)
        if len(i_a) > 0:
            def_imgs = split_img[i_a, :]
            n_normal = def_imgs.shape[0]
            #i_n = random.choices(i_n, k=n_normal*MTN)
            norm_imgs = split_img[i_n,:]
            test_lbls = np.append(np.full(len(def_imgs), 1), np.full(len(norm_imgs),0), axis = 0)
            test_imgs = np.append(def_imgs, norm_imgs)
            test_lbl_list = np.append(test_lbl_list, test_lbls)
            test_img_list = np.append(test_img_list, test_imgs)
    test_img_list = test_img_list.reshape(-1, 160,160)
    test_lbl_list = test_lbl_list.flatten()
    test_score = compute_loss(model,  test_img_list,test_lbl_list)
    pred = model.predict(test_img_list).flatten()
    print(pred[:10])
    print(test_lbl_list[:10])
    print('\nTesting score: ', test_score)
    tn, fp, fn, tp = confusion_matrix(test_lbl_list, np.round(pred)).ravel()
    print('tn, fp, fn, tp:  ', tn, fp, fn, tp)
    print('Number of defective images to train one epoch: ', defects)
    print('Number of defective images to test one epoch: ', test_defects)




