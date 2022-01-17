"""
01/01/2021

@author: Xin Huang
"""

import argparse
import glob
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from numpy import vstack
from numpy import argmax
from pickle import load
from data_utils import load_test_data
from data_utils import prepare_data
from model import Deep_coral
from train import evaluate_model_tgt

# _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # _device = torch.device('cpu')
# print("torch.cuda.is_available:")
# print(torch.cuda.is_available())
# # print("torch.cuda.get_device_name(0):")
# # print(torch.cuda.get_device_name(0))
# # import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# import numpy as np

# NUM_LALBELS = 6

# prefix = '/umbc/rs/nasa-access/data/calipso-viirs-merged/aerosol_free/'

# n_epochs_ddm = 20
# n_epochs = 20
# lambda_ = 0.001
# lambda_l2 = 0.05
# # NUM = 31
# DDM_NUM = 20
# # NUM = 26
# NUM = 22
# DIFFERECE_COL = 5
# BATCH_SIZE = 2048


# def prepare_data(X_src, Y_src, X_tgt, Y_tgt):
#     # load the train dataset
#     # src_train = CSVDataset(X_src, Y_src)
#     # tgt_train = CSVDataset(X_tgt, Y_tgt)
#     print("Y_src:")
#     print(Y_src)
#     print("Y_tgt:")
#     print(Y_tgt)
#     Y_src_1 = LabelEncoder().fit_transform(Y_src)
#     Y_tgt_1 = LabelEncoder().fit_transform(Y_tgt)

#     print("Y_src_1:")
#     print(Y_src_1)
#     print("Y_tgt_1:")
#     print(Y_tgt_1)

#     print("X_src:")
#     print(X_src)
#     print("X_tgt:")
#     print(X_tgt)

#     x_src = torch.from_numpy(X_src)
#     y_src = torch.from_numpy(Y_src_1)
#     x_tgt = torch.from_numpy(X_tgt)
#     y_tgt = torch.from_numpy(Y_tgt_1)

#     datasets = TensorDataset(x_src.float(), y_src, x_tgt.float(), y_tgt)
#     # prepare data loaders
#     train_dl = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)
#     return train_dl

# aggre_losses = []
# aggre_losses_l2 = []
# aggre_losses_coral = []
# aggre_losses_classifier = []
# aggre_losses_classifier_tgt = []

# aggre_losses_valid = []
# aggre_losses_l2_valid = []
# aggre_losses_classifier_valid = []
# aggre_losses_classifier_valid_tgt = []
# aggre_losses_coral_valid = []

# # aggre_losses_l2_ddm = []

# aggre_train_acc = []
# aggre_test_acc = []
# aggre_valid_acc = []
# aggre_train_tgt_acc = []

# evaluate the model target
# TODO: improve the predict function 
# def evaluate_model_tgt(test_dl, model, device):
#     model.eval()
#     predictions, actuals = list(), list()
#     # test_steps = len(test_dl)
#     # iter_test = iter(test_dl)
#     for i, (src_data, src_label, target_data, target_label) in enumerate(test_dl):
#     # for i in range(test_steps):
#         # evaluate the model on the test set
#         if torch.cuda.is_available():
#           target_data = target_data.to(device)
#           target_label = target_label.to(device)

#         tgt_data = target_data
#         targets = target_label
#         temp = torch.zeros((tgt_data.shape[0], DIFFERECE_COL))
#         temp = temp.to(device)
#         tmp_data = torch.cat((tgt_data, temp), 1)
#         # temp = torch.zeros((tgt_data.shape[0], DIFFERECE_COL))
#         # tmp_data = torch.cat((tgt_data, temp), 1)
#         with torch.no_grad():
#           _, yhat, _, _, _ = model(tmp_data, tgt_data)
#         # retrieve numpy array
#         yhat = yhat.detach().cpu().numpy()
#         actual = targets.cpu().numpy()
#         # convert to class labels
#         yhat = argmax(yhat, axis=1)
#         # reshape for stacking
#         actual = actual.reshape((len(actual), 1))
#         yhat = yhat.reshape((len(yhat), 1))
#         # store
#         predictions.append(yhat)
#         actuals.append(actual)
#     predictions, actuals = vstack(predictions), vstack(actuals)
#     # calculate accuracy
#     acc = accuracy_score(actuals, predictions)
#     return acc

# load the trained model and evaluate on the testing data
# model = Deep_coral(num_classes=NUM_LALBELS)
# model =  torch.load('/home/xinh1/access/ieee/'+'/model/2013to2017cpu.pth')
# # load the scaler
# from pickle import load
# sc_X = load(open('/home/xinh1/access/ieee' + '/model/scaler.pkl', 'rb'))

# def load_test_data(prefix, filename):
#   print("prefix:", prefix)
#   data_test = np.load(prefix + filename)

#   passive =1

#   #load common data
#   latlon_test = data_test['latlon']
#   # iff_test = data_test['iff']

#   # if passive ==1:
#   x_t_test = data_test['viirs']
#   y_t_test = data_test['label']
#   y_t_test = np.delete(y_t_test, 0, 1)
#   # else:
#   x_s_test = data_test['calipso']
#   y_s_test = data_test['label']
#   y_s_test = np.delete(y_s_test, 1 , 1)

#   inds_test,vals_test = np.where(y_t_test>0)

#   # process common data
#   Latlon_test = latlon_test[inds_test]
#   # Iff_test = iff_test[inds_test]

#   Y_t_test = y_t_test[inds_test]
#   X_t_test = x_t_test[inds_test]

#   Y_s_test = y_s_test[inds_test]
#   X_s_test = x_s_test[inds_test]

#   # 0 =< SZA <= 83
#   print('original X_t_test: ', X_t_test.shape)
#   rows_test = np.where((X_t_test[:,0] >= 0) & (X_t_test[:,0] <= 83) & (X_t_test[:,15] > 100) & (X_t_test[:,15] < 400) & (X_t_test[:,16] > 100) & (X_t_test[:,16] < 400) & (X_t_test[:,17] > 100) & (X_t_test[:,17] < 400) & (X_t_test[:,18] > 100) & (X_t_test[:,18] < 400) & (X_t_test[:,19] > 100) & (X_t_test[:,19] < 400) & (X_t_test[:,10] > 0))
#   print("rows_test:", rows_test)
#   print("rows_test.shape:", len(rows_test))

#   Latlon_test = Latlon_test[rows_test]
#   # Iff_test = Iff_test[rows_test]

#   Y_t_test = Y_t_test[rows_test]
#   X_t_test = X_t_test[rows_test]

#   Y_s_test = Y_s_test[rows_test]
#   X_s_test = X_s_test[rows_test]

#   X_s_test = np.nan_to_num(X_s_test)
#   X_t_test = np.nan_to_num(X_t_test)

#   print('after SZA X_t_test: ', X_t_test.shape)
#   print('after SZA X_s_test: ', X_s_test.shape)
#   X_s_test = np.concatenate((X_s_test, Latlon_test), axis=1)

#   print (X_s_test.shape)
#   print (X_t_test.shape)

#   X_test=np.concatenate((X_t_test, X_s_test), axis=1)

#   x_test2=sc_X.transform(X_test)

#   X_t_test = x_test2[:, 0:20]
#   x_test_c2 = x_test2[:, 20:45]
#   x_test_comm2 = x_test2[:, 45:47]

#   x_test_t_pt = X_t_test
#   print(x_test_t_pt.shape)

#   x_test_pt_test = np.concatenate((x_test_t_pt, x_test_comm2),axis=1)
#   print(x_test_pt_test.shape)

#   return X_s_test, Y_s_test, x_test_pt_test, Y_t_test

# all_results = []

# def test_all(prefix, filenames, device):
#   for i in range(len(filenames)):
#     X_s_test, Y_s_test, X_t_test, Y_t_test = load_test_data(prefix, filenames[i])

#     # X_t = x_train_pt
#     # Y_t = y_train

#     # # X_s_test = X_s_test
#     # # Y_s_test = Y_s_test
#     # X_t_test = x_test_pt_test
#     # Y_t_test = Y_s_test


#     test_dat = prepare_data(X_s_test, Y_s_test, X_t_test, Y_t_test)

#     # train_tgt, test_tgt = prepare_data(X_t, Y_t, X_t_test, Y_t_test)


#     # evaluate the model
#     acc = evaluate_model_tgt(test_dat, model, device)
#     all_results.append(acc)
#     print(filenames[i])
#     print('test Accuracy: %.3f' % acc)


# prefix = '/content/content/My Drive/Colab Notebooks/kddworkshop/fulldata/'
# test_filenames = ['test_138_day.npz', 'test_142_day.npz',  'test_144_day.npz', 'test_147_day.npz', 'test_154_day.npz', 'test_155_day.npz']
# test_filenames = ['2017_jan_day_005.npz', '2017_jan_day_013.npz',  '2017_jan_day_019.npz', '2017_jan_day_024.npz', '2017_jan_day_030.npz', '2017_mon1.npz', '2017.npz']
# test_all(prefix, test_filenames, _device)
# print("test_filenames:")
# print (test_filenames)
# print("all_results:")
# print(all_results)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--testing_data_path")
  parser.add_argument("--model_saving_path")
  args = parser.parse_args()

  NUM_LALBELS = 3
  BATCH_SIZE = 2048

  _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("torch.cuda.is_available:", torch.cuda.is_available())

  # load the trained model
  model = Deep_coral(num_classes=NUM_LALBELS)
  model = torch.load(args.model_saving_path + '/model.pth')
  sc_X = load(open(args.model_saving_path + '/scaler.pkl', 'rb'))

  # evaluate on the testing data
  test_files = glob.glob(args.testing_data_path + '/*.npz')
  for test_file in test_files:
    X_s_test, Y_s_test, X_t_test, Y_t_test = load_test_data(test_file, sc_X)
    test_dat = prepare_data(X_s_test, Y_s_test, X_t_test, Y_t_test, BATCH_SIZE)
    acc = evaluate_model_tgt(test_dat, model, device)
    print("Test on file:", test_file)
    print('Accuracy is: %.3f' % acc)








