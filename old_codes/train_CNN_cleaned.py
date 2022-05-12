import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from helpers.dataset_helpers import create_cnn_dataset
from old_codes.autoencoders import *
from common import *
from scipy.ndimage.interpolation import rotate
import random
import matplotlib.pyplot as plt
random.seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def classifier_scores(test_loss, train_loss):
    plt.plot(np.arange(0,len(test_loss)), test_loss, label = 'Test loss')
    plt.plot(np.arange(0,len(train_loss)), train_loss, linestyle= '--', label = 'Train loss')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Binary cross-entropy')
    plt.show()

def rotate_img(img):
    img = img.reshape(160,160)
    rot_angle = random.choice([90, 180, 270])
    rot = rotate(img, rot_angle)
    return rot.flatten()

def change_bright_img(img):
    img = img.flatten()
    change = 0.5
    img = np.multiply(img, change)
    return img

#def split_img(item):
#    tf.image.extract_patches(images=item,sizes=[1, 160, 160, 1],strides=[1, 160, 160, 1], rates = [1,1,1,1], padding='VALID')

def encode_split(img):
    img = img[0].numpy().reshape(1, 2736, 3840, 1)[:, :2720, :, :]
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = np.sqrt((decoded_img - img)**2)
    split_img = tf.image.extract_patches(images=aed_img, sizes=[1, 160, 160, 1], strides=[1, 160, 160, 1],rates=[1, 1, 1, 1], padding='VALID')
    return split_img.numpy().reshape(17 * 24, 160 * 160)

bce = tf.keras.losses.BinaryCrossentropy()
@tf.function
def compute_loss_train(model, x, y_ref):
    y = model(x, training=True)
    reconstruction_error = bce(y_ref, y)
    return reconstruction_error

@tf.function
def compute_loss_test(model, x, y_ref):
    y = model(x, training=False)
    reconstruction_error = bce(y_ref, y)
    return reconstruction_error

# define the training step
@tf.function
def train_step(model, x, y_ref, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss_train(model, x, y_ref)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def plot_img_list(def_imgs, i):
    def_imgs = def_imgs[:4]
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(def_imgs[0].reshape(160,160), vmin=0, vmax=255)
    axs[0, 1].imshow(def_imgs[1].reshape(160,160), vmin=0, vmax=255)
    axs[1, 0].imshow(def_imgs[2].reshape(160,160), vmin=0, vmax=255)
    axs[1, 1].imshow(def_imgs[3].reshape(160,160), vmin=0, vmax=255)
    plt.tight_layout()
    #plt.savefig('aed_imgs/subimgs_%i.png' % i, dpi=600)
    plt.show()

def plot_normimg_list(def_imgs, i):
    def_imgs = def_imgs[:4]
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(def_imgs[0].reshape(160,160))
    axs[0, 1].imshow(def_imgs[1].reshape(160,160))
    axs[1, 0].imshow(def_imgs[2].reshape(160,160))
    axs[1, 1].imshow(def_imgs[3].reshape(160,160))
    plt.tight_layout()
    #plt.savefig('aed_imgs/norm_subimgs_%i.png' % i, dpi=600)
    plt.show()

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

train_def_dataset = train_def_dataset.shuffle(N_det_train, seed=1, reshuffle_each_iteration=False).take(400)
test_def_dataset = test_def_dataset.shuffle(N_det_test, seed= 1, reshuffle_each_iteration=False).take(80)

ae = AutoEncoder()
ae.load('/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/TQ3_1_cont/model_AE_TQ3_404_to_404_epochs')

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(160,160,1)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(4,4)  , padding="valid",activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(4,4) , padding="valid" ,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(4, 4), padding="valid", activation='relu',),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), activation = 'sigmoid', padding="valid"),
    tf.keras.layers.Flatten(),
])

savename = 'does_overfit'

#model = tf.keras.models.load_model('saved_class/saved_cnn_3')
print(model.summary())

optimizer = tf.keras.optimizers.Adam(1e-4)

epochs = 200
#how many normal images for each defective
MTN = 5
batch_size = 64

test_scores, train_scores = [], []
for epoch in range(epochs):
    random.seed(42)
    np.random.seed(42)
    print("\nStart of epoch %d" % (epoch,))
    defects, test_defects, plot = 0, 0, 0
    train_score_epoch = []

    train_lbl_list, train_img_list = [], []
    for X_train, Y_train in tqdm(train_def_dataset, total=tf.data.experimental.cardinality(train_def_dataset).numpy()):
        Y_train = Y_train.numpy().reshape(17 * 24)
        i_a = np.where(Y_train == 1)[0]
        i_n = np.where(Y_train == 0)[0]
        split_img = encode_split(X_train)
        defects = defects + len(i_a)
        if len(i_a) > 0:
            def_imgs = split_img[i_a, :]
            rotated = np.array([rotate_img(item) for item in def_imgs])
            rotated_imgs = np.append(def_imgs, rotated, axis = 0)
            #bright_imgs = np.array([change_bright_img(item) for item in rotated_imgs])
            if epoch < 2 and len(i_a) == 4 and plot < 4:
                plot_img_list(def_imgs, defects)
                plot_img_list(rotated, defects)
            #    plot_img_list(bright_imgs, defects)
            #def_imgs = np.append(rotated_imgs, bright_imgs, axis = 0)
            def_imgs = rotated_imgs
            n_normal = def_imgs.shape[0]
            i_n = np.random.choice(a=i_n, size=n_normal*MTN, replace = False)
            norm_imgs = split_img[i_n, :]
            if epoch < 2 and len(i_a) == 4 and plot < 4:
                plot_normimg_list(norm_imgs[:4], defects)
                plot = plot + 1
            train_lbls = np.append(np.full(len(def_imgs), 1), np.full(len(norm_imgs),0), axis = 0)
            train_imgs = np.append(def_imgs, norm_imgs, axis = 0).reshape(-1, 160, 160)
            train_lbl_list = np.append(train_lbl_list, train_lbls)
            train_img_list = np.append(train_img_list, train_imgs)
    print('Number of training samples: ', len(train_img_list), len(train_img_list))
    print('Number of defective images to train one epoch: ', defects)
    train_dataset_epoch = tf.data.Dataset.from_tensor_slices((train_img_list, train_lbl_list)).shuffle(100).batch(batch_size)
    for x_batch, y_batch in train_dataset_epoch:
        train_step(model, x_batch, y_batch, optimizer)
    train_score = compute_loss_test(model, train_img_list, train_lbl_list)
    print('Training loss: ', train_score, axis = 0)
    train_scores = np.append(train_scores, train_score)
    model.save('saved_class/saved_cnn_%s' % savename)

    print('\nStarting testing:')
    test_lbl_list, test_img_list = [], []
    for X_test, Y_test in tqdm(test_def_dataset, total=tf.data.experimental.cardinality(test_def_dataset).numpy()):
        Y_test = Y_test.numpy().reshape(17*24)
        i_a = np.where(Y_test == 1)[0]
        i_n = np.where(Y_test == 0)[0]
        split_img = encode_split(X_test)
        test_defects = test_defects + len(i_a)
        if len(i_a) > 0:
            def_imgs = split_img[i_a, :]
            n_normal = def_imgs.shape[0]
            i_n = np.random.choice(a=i_n, size=n_normal*MTN, replace = False)
            norm_imgs = split_img[i_n,:]
            test_lbls = np.append(np.full(len(def_imgs), 1), np.full(len(norm_imgs),0), axis = 0)
            test_imgs = np.append(def_imgs, norm_imgs)
            test_lbl_list = np.append(test_lbl_list, test_lbls)
            test_img_list = np.append(test_img_list, test_imgs)
    print('Number of testing samples: ', len(test_img_list), len(test_img_list))
    print('Number of defective images to test one epoch: ', test_defects)
    test_img_list = test_img_list.reshape(-1, 160,160)
    test_lbl_list = test_lbl_list.flatten()
    test_score = compute_loss_test(model, test_img_list, test_lbl_list)
    test_scores = np.append(test_scores, test_score)
    test_pred = model.predict(test_img_list).flatten()
    print(test_pred[:5])
    print(test_lbl_list[:5])
    print('Testing score: ', test_score)
    tn, fp, fn, tp = confusion_matrix(test_lbl_list, np.round(test_pred)).ravel()
    print('Test tn, fp, fn, tp:  ', tn, fp, fn, tp)
    classifier_scores(test_scores, train_scores)
    np.save('losses/test_loss_%s.npy' % savename, test_scores)
    np.save('losses/train_loss_%s.npy' % savename, train_scores)