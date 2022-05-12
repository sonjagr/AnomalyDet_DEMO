import numpy as np

base_dir = 'db/'
dir_ae = "AE/"
dir_det = 'DET/'

X_train_ae_list = np.load(base_dir + dir_ae + 'X_train_AE.npy', allow_pickle=True)
X_test_ae_list = np.load(base_dir + dir_ae + 'X_test_AE.npy', allow_pickle=True)

X_train_det_list = np.load(base_dir + dir_det + 'X_train_DET.npy', allow_pickle=True)
X_test_det_list = np.load(base_dir + dir_det + 'X_test_DET.npy', allow_pickle=True)

