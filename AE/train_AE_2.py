import numpy as np
import tensorflow
import math, time
import pickle, argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from helpers.dataset_helpers import create_dataset, resize
from AE.autoencoders2 import *
from common import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_ID", type=str, help="Model ID",
                    default="TQ1", required=True)
parser.add_argument("--batch_size", type=int,
                    help="Batch size for the training", default=8, required=False)
parser.add_argument("--epochs", type=int,
                    help="Number of epochs for the training", default=40, required=False)
parser.add_argument("--gpu", type=str,
                    help="Which gpu to use", default="1", required=False)
parser.add_argument("--savename", type=str,
                    help="Name to save with", required=True)
parser.add_argument("--load", type=bool,
                    help="Load old model or not", default = False, required=False)
parser.add_argument("--lr", type=float,
                    help="Lr", default = 1e-4, required=False)
parser.add_argument("--contfromepoch", type=str,
                    help="If load = True: Epoch to continue training from", default=1, required=False)
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
model_ID = args.model_ID
gpu = args.gpu
savename = args.savename
load = args.load
lr = args.lr
cont_epoch = args.contfromepoch
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

@tensorflow.function
def L2_loss_AE(model, x):
    x_encoded = model.encode(x)
    x_decoded = model.decode(x_encoded)
    x = tensorflow.cast(x, tensorflow.float32)
    reconstruction_error = tensorflow.reduce_mean(tensorflow.square(tensorflow.subtract(x_decoded, x)))
    return reconstruction_error

@tensorflow.function
def L1_loss_AE(model, x):
    x_encoded = model.encode(x)
    x_decoded = model.decode(x_encoded)
    x = tensorflow.cast(x, tensorflow.float32)
    reconstruction_error = tensorflow.reduce_mean(tensorflow.abs(tensorflow.subtract(x_decoded, x)))
    return reconstruction_error

@tensorflow.function
def train_step(model, x, optimizer):
    with tensorflow.GradientTape() as tape:
        loss = L2_loss_AE(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def Callback_EarlyStopping(LossList, min_delta=0.1, patience=20):
    if len(LossList)//patience < 2:
        return False
    mean_previous = np.mean(LossList[::-1][patience:2*patience])
    mean_recent = np.mean(LossList[::-1][:patience])
    delta_abs = np.abs(mean_recent - mean_previous) #abs change
    delta_abs = np.abs(delta_abs / mean_previous)  # relative change
    if delta_abs < min_delta :
        print("CallBack: Loss didn't change much from last %d epochs"%(patience))
        print("CallBack: Percent change in loss value:", delta_abs*1e2)
        return True
    else:
        return False

base_dir = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/db/'
dir_ae = "/"
X_train_list = np.load(os.path.join(base_dir, dir_ae, 'X_train_AE.npy'), allow_pickle=True)
X_test_list = np.load(os.path.join(base_dir, dir_ae, 'X_test_AE.npy'), allow_pickle=True)

np.random.seed(1)
X_train_list = np.random.choice(X_train_list, 6000, replace=False)
X_test_val_list = np.random.choice(X_test_list, 2000, replace=False)
X_test_list, X_val_list = train_test_split(X_test_val_list, test_size=0.5, random_state=42)

X_train_list = [imgDir_gpu + s for s in X_train_list]
X_test_list = [imgDir_gpu + s for s in X_test_list]
X_val_list = [imgDir_gpu + s for s in X_val_list]

train_dataset = create_dataset(X_train_list, _shuffle=True).batch(batch_size)
test_dataset = create_dataset(X_test_list, _shuffle=True).batch(batch_size)
val_dataset = create_dataset(X_val_list, _shuffle=True).batch(batch_size)

train_dataset = train_dataset.map(lambda item: tuple(tf.py_function(resize, [item], [tf.float32,])))
test_dataset = test_dataset.map(lambda item: tuple(tf.py_function(resize, [item], [tf.float32,])))

N_train = len(X_train_list)
N_test = len(X_test_list)

ae = AutoEncoder()
if load == False:
    if model_ID == 'TQ1':
        ae.initialize_network_TQ1()
    elif model_ID == 'TQ2':
        ae.initialize_network_TQ2()
    elif model_ID == 'TQ3':
        ae.initialize_network_TQ3()
    elif model_ID == 'TQ3_2':
        ae.initialize_network_TQ3_2()
    elif model_ID == 'TQ3_3':
        ae.initialize_network_TQ3_3()
    else:
        print('No model defined named ', model_ID)
    min_steps_train = []
    train_costs = []
    min_steps_test = []
    test_costs = []
    start_epoch = 0

cp_loc = '/afs/cern.ch/user/s/sgroenro/anomaly_detection/checkpoints/'
if load == True:
    ae.load(cp_loc + model_ID + '_' + str(batch_size) + '_' + str(savename) + '/AE_' + model_ID + '_' + str(cont_epoch) + '_to_' + str(cont_epoch) + '_epochs')
    with open(cp_loc + model_ID + '_' + str(batch_size) + '_' + str(savename) + '/cost.pkl', "rb") as cost_file_pkl:
        _x = pickle.load(cost_file_pkl)
        min_steps_train = _x["min_steps_train"]
        train_costs = _x["train_losses"]
        min_steps_test = _x["min_steps_test"]
        test_costs = _x["test_losses"]
        start_epoch = int(cont_epoch) + 1

patience = 10
delta = 0.01
saveModelEvery = 1

model_name = 'AE_'+model_ID
save_location = os.path.join(TrainDir_gpu, '../checkpoints', model_ID + '_' + str(batch_size) + '_' + str(savename))
Path(save_location).mkdir(parents=True, exist_ok=True)
costFigurePath = os.path.join(save_location, "cost.pdf")
cost_file_path = costFigurePath.replace(".pdf", ".pkl")

optimizer = tensorflow.keras.optimizers.Adam(lr)
N_per_epoch = math.ceil(N_train / batch_size)
min_step = 0 if len(min_steps_train) == 0 else min_steps_train[-1]
if load == False:
    start_epoch = int(min_step / N_per_epoch)

for epoch in range(start_epoch, epochs + 1):
    print('Epoch: {} started'.format(epoch))
    start_time = time.time()
    for train_x in tqdm(train_dataset, total=tensorflow.data.experimental.cardinality(train_dataset).numpy()):
        min_step += 1
        if min_step == 1:
            print(tf.shape(train_x))
        train_step(ae, train_x, optimizer)
        loss = tensorflow.keras.metrics.Mean()
        loss(L2_loss_AE(ae, train_x))
        L2 = loss.result()
        train_costs.append(L2.numpy())
        min_steps_train.append(min_step)

    print("Training sequence ended. Testing initiated.")
    loss = tensorflow.keras.metrics.Mean()
    for test_x in tqdm(test_dataset, total=tensorflow.data.experimental.cardinality(test_dataset).numpy()):
        loss(L2_loss_AE(ae, test_x))
    test_loss = loss.result().numpy()
    min_steps_test.append(min_step)
    test_costs.append(test_loss)

    stopEarly = Callback_EarlyStopping(test_costs, min_delta=0.1, patience=20)
    stopEarly = False
    if stopEarly:
        print(f'\nEarly stopping. No improvement of more than {delta:.5%} in '
              f'validation loss in the last {patience} epochs.')
        print("Terminating training ")
        break

    end_time = time.time()
    print(f'Epoch: {epoch}, Test set L2-loss: {test_loss}, time elapsed for current epoch: {round(end_time - start_time, 1)} sec')

    if os.path.exists(cost_file_path):
        os.remove(cost_file_path)
    with open(cost_file_path, "wb") as cost_file:
        pickle.dump({
            "train_losses": train_costs,
            "test_losses": test_costs,
            "min_steps_train": min_steps_train,
            "min_steps_test": min_steps_test
        }, cost_file)

    lower_bound = int(epoch / saveModelEvery) * saveModelEvery
    upper_bound = int(epoch / saveModelEvery) * \
                  saveModelEvery + saveModelEvery - 1
    _currentModelFile = "%s_%i_to_%i_epochs" % (model_name, lower_bound, upper_bound)
    _currentModelFile = os.path.join(save_location, _currentModelFile)
    if os.path.exists(_currentModelFile):
        shutil.rmtree(_currentModelFile)
    ae.save(_currentModelFile)
    print("Model saved to", _currentModelFile)

print('Training and testing finished')
