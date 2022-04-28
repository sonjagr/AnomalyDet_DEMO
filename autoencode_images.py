import numpy as np
import tensorflow
import os, sys, math, time
import pickle, argparse
import shutil
from pathlib import Path
from tensorflow import keras
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from helpers.dataset_helpers import create_dataset, create_cnn_dataset
from autoencoders import *
from common import *

base_dir = 'db/'
dir_ae = "AE/"
dir_det = 'DET/'

X_train_ae_list = np.load(base_dir + dir_ae + 'X_train_AE.npy', allow_pickle=True)
X_test_ae_list = np.load(base_dir + dir_ae + 'X_test_AE.npy', allow_pickle=True)

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)

