from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import glob
from pickle import dump
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch

def equalRate(a, b):
  c = a-b
  d = np.where(c==0)
  print("equal labels:")
  print(d[0].shape[0] * 1.0 / a.shape[0])

def loadData(filename):
    data = np.load(filename)

    # Load all relevant fields
    latlon = data['latlon']
    iff = data['iff']
    X_v = data['viirs']
    Y_v = np.delete(data['label'], 0, 1)  # VIIRS label = column 1
    X_c = data['calipso']
    Y_c = np.delete(data['label'], 1, 1)  # CALIPSO label = column 0

    # Remap classes 1 and 2 → 1 (Cloud), keep 0 (Clear)
    Y_v[Y_v == 2] = 1
    Y_v[Y_v == 1] = 1
    Y_c[Y_c == 2] = 1
    Y_c[Y_c == 1] = 1

    equalRate(data['label'][:, 0], data['label'][:, 1])

    # Only keep valid label indices (> -1)
    inds_v, _ = np.where(Y_v >= 0)
    X_v = X_v[inds_v]
    Y_v = Y_v[inds_v]
    Latlon = latlon[inds_v]
    Iff = iff[inds_v]

    inds_c = inds_v  # synced indexing
    X_c = X_c[inds_c]
    Y_c = Y_c[inds_c]

    # Filter based on solar zenith angle and radiance sanity check
    rows = np.where(
        (X_v[:, 0] >= 0) & (X_v[:, 0] <= 83) &  # SZA
        (X_v[:, 10] > 0) &  # VZA
        (X_v[:, 15] > 100) & (X_v[:, 15] < 400) &
        (X_v[:, 16] > 100) & (X_v[:, 16] < 400) &
        (X_v[:, 17] > 100) & (X_v[:, 17] < 400) &
        (X_v[:, 18] > 100) & (X_v[:, 18] < 400) &
        (X_v[:, 19] > 100) & (X_v[:, 19] < 400)
    )[0]

    X_v = X_v[rows]
    Y_v = Y_v[rows]
    X_c = X_c[rows]
    Y_c = Y_c[rows]
    Latlon = Latlon[rows]
    Iff = Iff[rows]

    # Add lat/lon to CALIPSO input
    X_c = np.concatenate((X_c, Latlon), axis=1)

    # Replace NaNs with 0
    X_v = np.nan_to_num(X_v)
    X_c = np.nan_to_num(X_c)

    return X_v, X_c, Y_v, Y_c


def load_test_data(filename, sc_X):
    data_test = np.load(filename)

    latlon_test = data_test['latlon']
    x_t_test = data_test['viirs']
    y_t_test = np.delete(data_test['label'], 0, 1)
    x_s_test = data_test['calipso']
    y_s_test = np.delete(data_test['label'], 1, 1)

    # Remap labels: 1 and 2 → 1 (Cloud)
    y_t_test[y_t_test == 2] = 1
    y_t_test[y_t_test == 1] = 1
    y_s_test[y_s_test == 2] = 1
    y_s_test[y_s_test == 1] = 1

    # Only keep valid target indices
    inds_test, _ = np.where(y_t_test >= 0)
    X_t_test = x_t_test[inds_test]
    Y_t_test = y_t_test[inds_test]
    X_s_test = x_s_test[inds_test]
    Y_s_test = y_s_test[inds_test]
    Latlon_test = latlon_test[inds_test]

    # Filter based on SZA and radiance range
    rows_test = np.where(
        (X_t_test[:, 0] >= 0) & (X_t_test[:, 0] <= 83) &
        (X_t_test[:, 10] > 0) &
        (X_t_test[:, 15] > 100) & (X_t_test[:, 15] < 400) &
        (X_t_test[:, 16] > 100) & (X_t_test[:, 16] < 400) &
        (X_t_test[:, 17] > 100) & (X_t_test[:, 17] < 400) &
        (X_t_test[:, 18] > 100) & (X_t_test[:, 18] < 400) &
        (X_t_test[:, 19] > 100) & (X_t_test[:, 19] < 400)
    )[0]

    X_t_test = X_t_test[rows_test]
    Y_t_test = Y_t_test[rows_test]
    X_s_test = X_s_test[rows_test]
    Y_s_test = Y_s_test[rows_test]
    Latlon_test = Latlon_test[rows_test]

    # Replace NaNs with 0 and append lat/lon
    X_t_test = np.nan_to_num(X_t_test)
    X_s_test = np.nan_to_num(X_s_test)
    X_s_test = np.concatenate((X_s_test, Latlon_test), axis=1)

    # Combine both for scaling
    X_test = np.concatenate((X_t_test, X_s_test), axis=1)
    X_test_scaled = sc_X.transform(X_test)

    X_t_scaled = X_test_scaled[:, 0:20]
    X_c_scaled = X_test_scaled[:, 20:45]
    X_common_scaled = X_test_scaled[:, 45:47]

    # Final preprocessed input for target (VIIRS + lat/lon)
    X_tgt_final = np.concatenate((X_t_scaled, X_common_scaled), axis=1)

    return X_s_test, Y_s_test, X_tgt_final, Y_t_test


def prepare_data(X_src, Y_src, X_tgt, Y_tgt, b_size=2048):

    Y_src_1 = LabelEncoder().fit_transform(Y_src)
    Y_tgt_1 = LabelEncoder().fit_transform(Y_tgt)

    x_src = torch.from_numpy(X_src)
    y_src = torch.from_numpy(Y_src_1)
    x_tgt = torch.from_numpy(X_tgt)
    y_tgt = torch.from_numpy(Y_tgt_1)

    datasets = TensorDataset(x_src.float(), y_src, x_tgt.float(), y_tgt)
    # prepare data loaders
    train_dl = DataLoader(datasets, batch_size=b_size, shuffle=True)
    return train_dl

def prepare_data_predict(X_tgt, b_size=2048):
    # load the train dataset
    x_tgt = torch.from_numpy(X_tgt)
    # y_tgt = torch.from_numpy(Y_tgt_1)

    datasets = TensorDataset(x_tgt.float())
    # prepare data loaders
    train_dl = DataLoader(datasets, batch_size=b_size, shuffle=False)
    return train_dl

def preProcessing(training_data_path, model_saving_path, b_size):
  # load the training data
  prefix = training_data_path

  file_count = 0
  train_files = glob.glob(prefix + '/*.npz')
  for train_file in train_files:
    if file_count < 1:
      X_v_tmp, X_c_tmp, Y_v_tmp, Y_c_tmp  = loadData(train_file)
      file_count += 1
    else:
       X_v_1, X_c_1, Y_v_1, Y_c_1 = loadData(train_file)
       X_v_tmp = np.concatenate((X_v_tmp, X_v_1), axis = 0)
       X_c_tmp = np.concatenate((X_c_tmp, X_c_1), axis = 0)
       Y_v_tmp = np.concatenate((Y_v_tmp, Y_v_1), axis=0)
       Y_c_tmp = np.concatenate((Y_c_tmp, Y_c_1), axis=0)
       del X_v_1, X_c_1, Y_v_1, Y_c_1
       file_count += 1

  X_v = X_v_tmp
  X_c = X_c_tmp
  Y_v = Y_v_tmp
  Y_c = Y_c_tmp
  del X_v_tmp, X_c_tmp, Y_v_tmp, Y_c_tmp

  Y = np.concatenate((Y_v, Y_c), axis=1)
  print ("X_v.shape:", X_v.shape)
  print ("X_c.shape:", X_c.shape)
  print ("Y.shape:", Y.shape)

  # combine data and split latter to define ground truth for MLR
  X=np.concatenate((X_v, X_c), axis=1)
  print (X.shape)
  x_train, x_valid, y_train, y_valid = train_test_split(X, Y,
                                                      test_size=0.3,
                                                      random_state=0,
                                                      stratify=Y)

  del X, Y, X_v, X_c, Y_v, Y_c

  # feature scaling
  sc_X = StandardScaler()
  x_train=sc_X.fit_transform(x_train)
  x_valid=sc_X.transform(x_valid)

  # save the scaler
  dump(sc_X, open(model_saving_path + '/scalerCM.pkl', 'wb'))
  print("saved scaler done")

  x_train_v = x_train[:, 0:20]
  x_train_c = x_train[:, 20:45]
  x_train_comm = x_train[:, 45:47]
  x_train_src = x_train[:, 20:47]
  y_train_src = np.delete(y_train, 0, 1)
  y_train_tgt = np.delete(y_train, 1, 1)

  x_valid_src = x_valid[:, 20:51]
  x_valid_v = x_valid[:, 0:20]
  x_valid_c = x_valid[:, 20:45]
  x_valid_comm = x_valid[:, 45:47]
  y_valid_src = np.delete(y_valid, 0, 1)
  y_valid_tgt = np.delete(y_valid, 1, 1)

  x_train_c_pt = x_train_v
  x_valid_c_pt = x_valid_v
  print(x_train_c_pt.shape)
  print(x_valid_c_pt.shape)

  # DLR imputed target domain
  x_train_pt = np.concatenate((x_train_c_pt, x_train_comm),axis=1)
  print(x_train_pt.shape)

  x_valid_pt = np.concatenate((x_valid_c_pt, x_valid_comm),axis=1)
  print(x_valid_pt.shape)

  # train data
  X_s = x_train_src
  Y_s = y_train_src
  X_t = x_train_pt
  Y_t = y_train_tgt

  # valid data
  X_s_valid = x_valid_src
  Y_s_valid = y_valid_src
  X_t_valid = x_valid_pt
  Y_t_valid = y_valid_tgt

  train_dat = prepare_data(X_s, Y_s, X_t, Y_t, b_size)
  valid_dat = prepare_data(X_s_valid, Y_s_valid, X_t_valid, Y_t_valid, b_size)

  return train_dat, valid_dat

