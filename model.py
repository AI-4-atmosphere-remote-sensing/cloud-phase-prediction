"""
Cloud phase prediction 
01/01/2021

@@author: Xin Huang
"""

from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Softmax
from torch.nn import Module
from torch.nn import Dropout
from torch.nn import BatchNorm1d
from torch.optim import SGD,RMSprop,Adam
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.utils.data import TensorDataset
import argparse
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import glob
from pickle import dump
import pandas as pd


def equalRate(a, b):
  c = a-b
  d = np.where(c==0)
  print("equal labels:")
  print(d[0].shape[0] * 1.0 / a.shape[0])

def loadData(filename): 
  data = np.load(filename)
  # passive = 1
  #load common data
  latlon = data['latlon']
  iff = data['iff']

  X_v = data['viirs']
  Y_v = data['label']
  print ('X_v shape:')
  print (X_v.shape)
  print ('Y_v.shape:')
  print(Y_v)
  Y_v = np.delete(Y_v, 0, 1)
  print ('Y_v.shape:')


  X_c = data['calipso']
  Y_c = data['label']
  print ('X_c shape:')
  print (X_c.shape)
  print ('Y_c.shape:')
  Y_c = np.delete(Y_c, 1, 1)
  print ('Y_c.shape:')

  equalRate(data['label'][:,0], data['label'][:,1])

  inds_v,vals_v = np.where(Y_v>0)
  Y_v = Y_v[inds_v]
  X_v = X_v[inds_v]
  print ('X_v')
  print (X_v)

  inds_c,vals_c = np.where(Y_v>0)
  Y_c = Y_c[inds_c]
  X_c = X_c[inds_c]
  print ('X_c')
  print (X_c)

  # process common data
  Latlon = latlon[inds_v]
  Iff = iff[inds_v]

  print('original X_v: ', X_v.shape)
  rows = np.where((X_v[:,0] >= 0) & (X_v[:,0] <= 83) & (X_v[:,15] > 100) & (X_v[:,15] < 400) & (X_v[:,16] > 100) & (X_v[:,16] < 400) & (X_v[:,17] > 100) & (X_v[:,17] < 400) & (X_v[:,18] > 100) & (X_v[:,18] < 400) & (X_v[:,19] > 100) & (X_v[:,19] < 400) & (X_v[:,10] > 0))
  print("rows:", rows)
  print("rows.shape:", len(rows))

  Latlon = Latlon[rows]
  Iff = Iff[rows]
  print("Iff:", Iff.shape)
  # Iff = Iff[:, 0]
  # print("Iff0:", Iff.shape)

  Y_v = Y_v[rows]
  X_v = X_v[rows]

  Y_c = Y_c[rows]
  X_c = X_c[rows]

  print('after SZA X_v: ', X_v.shape)
  print('after SZA X_c: ', X_c.shape)

  #concanate common data
  # X_v = np.concatenate((X_v, Latlon, Iff), axis=1)
  # X_c = np.concatenate((X_c, Latlon, Iff), axis=1)
  X_c = np.concatenate((X_c, Latlon), axis=1)
  print (X_v.shape)
  print (X_c.shape)

  X_v = np.nan_to_num(X_v)
  X_c = np.nan_to_num(X_c)
  return X_v, X_c, Y_v, Y_c



def load_test_data(prefix, filename):
  data_test = np.load(prefix + filename)

  passive =1

  #load common data
  latlon_test = data_test['latlon']
  # iff_test = data_test['iff']

  # if passive ==1:
  x_t_test = data_test['viirs']
  y_t_test = data_test['label']
  y_t_test = np.delete(y_t_test, 0, 1)
  # else:
  x_s_test = data_test['calipso']
  y_s_test = data_test['label']
  y_s_test = np.delete(y_s_test, 1 , 1)

  inds_test,vals_test = np.where(y_t_test>0)

  # process common data
  Latlon_test = latlon_test[inds_test]
  # Iff_test = iff_test[inds_test]

  Y_t_test = y_t_test[inds_test]
  X_t_test = x_t_test[inds_test]

  Y_s_test = y_s_test[inds_test]
  X_s_test = x_s_test[inds_test]

  # 0 =< SZA <= 83
  print('original X_t_test: ', X_t_test.shape)
  rows_test = np.where((X_t_test[:,0] >= 0) & (X_t_test[:,0] <= 83) & (X_t_test[:,15] > 100) & (X_t_test[:,15] < 400) & (X_t_test[:,16] > 100) & (X_t_test[:,16] < 400) & (X_t_test[:,17] > 100) & (X_t_test[:,17] < 400) & (X_t_test[:,18] > 100) & (X_t_test[:,18] < 400) & (X_t_test[:,19] > 100) & (X_t_test[:,19] < 400) & (X_t_test[:,10] > 0))
  print("rows_test:", rows_test)
  print("rows_test.shape:", len(rows_test))

  Latlon_test = Latlon_test[rows_test]
  # Iff_test = Iff_test[rows_test]

  Y_t_test = Y_t_test[rows_test]
  X_t_test = X_t_test[rows_test]

  Y_s_test = Y_s_test[rows_test]
  X_s_test = X_s_test[rows_test]

  X_s_test = np.nan_to_num(X_s_test)
  X_t_test = np.nan_to_num(X_t_test)

  print('after SZA X_t_test: ', X_t_test.shape)
  print('after SZA X_s_test: ', X_s_test.shape)
  X_s_test = np.concatenate((X_s_test, Latlon_test), axis=1)

  print (X_s_test.shape)
  print (X_t_test.shape)

  X_test=np.concatenate((X_t_test, X_s_test), axis=1)

  x_test2=sc_X.transform(X_test)

  X_t_test = x_test2[:, 0:20]
  x_test_c2 = x_test2[:, 20:45]
  x_test_comm2 = x_test2[:, 45:47]

  x_test_t_pt = X_t_test
  print(x_test_t_pt.shape)

  x_test_pt_test = np.concatenate((x_test_t_pt, x_test_comm2),axis=1)
  print(x_test_pt_test.shape)

  return X_s_test, Y_s_test, x_test_pt_test, Y_t_test



# run the Correlation based DA
# pytorch mlp for multiclass classification

#n_epochs_ddm = 20


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
                
# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, X1, Y1):
        # load the csv file as a dataframe
        self.X=X1
        self.y=Y1
        print("self.X before fit_transform")
        print(self.X)
        print("self.y before fit_transform")
        print(self.y)
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        print("self.X before fit_transform")
        print(self.X)
        print("self.y after fit_transform")
        print(self.y)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

def CORAL(src,tgt):
    d = src.size(1)
    src_c = coral(src)
    tgt_c = coral(tgt)

    loss = torch.sum(torch.mul((src_c-tgt_c),(src_c-tgt_c)))
    loss = loss/(4*d*d)
    return loss

def LOG_CORAL(src,tgt):
    d = src.size(1)
    src_c = coral(src)
    tgt_c = coral(tgt)
    src_vals, src_vecs = torch.symeig(src_c,eigenvectors = True)
    tgt_vals, tgt_vecs = torch.symeig(tgt_c,eigenvectors = True)
    src_cc = torch.mm(src_vecs,torch.mm(torch.diag(torch.log(src_vals)),src_vecs.t()))
    tgt_cc = torch.mm(tgt_vecs,torch.mm(torch.diag(torch.log(tgt_vals)),tgt_vecs.t()))
    loss = torch.sum(torch.mul((src_cc - tgt_cc), (src_cc - tgt_cc)))
    loss = loss / (4 * d * d)
    return loss


def coral(data):
    n = data.size(0)
    id_row = torch.ones(n).resize(1,n)
    if torch.cuda.is_available():
        id_row = id_row.cuda()
    sum_column = torch.mm(id_row,data)
    mean_column = torch.div(sum_column,n)
    mean_mean = torch.mm(mean_column.t(),mean_column)
    d_d = torch.mm(data.t(),data)
    coral_result = torch.add(d_d,(-1*mean_mean))*1.0/(n-1)
    return coral_result

class Deep_coral(Module):
    def __init__(self,num_classes = 3):
        super(Deep_coral,self).__init__()
        #NUM:26 => 22; DIFFERECE_COL = 5; DDM_NUM = 20; DIFFERECE_COL=5
        self.DDM_NUM = 20
        self.DIFFERECE_COL = 5
        self.NUM = 22
        self.ddm = DDM(n_inputs=self.DDM_NUM,n_outputs=self.DDM_NUM+self.DIFFERECE_COL)
        self.feature = MLP(n_inputs=self.NUM + self.DIFFERECE_COL)
        self.central = Linear(64,32) # correlation layer
        xavier_uniform_(self.central.weight)

        self.fc = CLASSIFY(32, num_classes)
        # self.fc = Linear(32,num_classes)
        # xavier_uniform_(self.fc.weight)

        #  initial layer
        # self.init_layer = Linear(NUM+5, NUM)
        # xavier_uniform_(self.init_layer.weight)
        # self.act3 = Softmax(dim=1)
        # self.fc.weight.data.normal_(0,0.005)# initialization

        # temp = torch.zeros((tgt_data.shape[0], DIFFERECE_COL))
        # tmp_data = torch.cat((tgt_data, temp), 1)
    def forward(self,src,tgt):
        src = self.feature(src)
        centr1 = self.central(src)
        src = self.fc(centr1)
        # output layer
        viirs_d = tgt[:, 0:20]
        common_d = tgt[:, 20:self.NUM] #NUM = 26>22
        dmval = self.ddm(viirs_d)
        combine_d = torch.cat((dmval, common_d), 1)
        tgt = self.feature(combine_d)
        centr2 = self.central(tgt)
        # tgt = self.feature(tgt)
        tgt = self.fc(centr2)
        return src,tgt,dmval,centr1,centr2

    def pretrain(self,tgt):
        # output layer
        print("tgt:", tgt.shape)
        print(tgt)
        viirs_d = tgt[:, 0:20]
        common_d = tgt[:, 20:self.NUM]
        # dmval = self.ddm(tgt)
        dmval = self.ddm(viirs_d)
        # combine_d = torch.cat((dmval, common_d), 1)
        # tgt = self.feature(combine_d)
        # tgt = self.fc(tgt)
        return dmval

    def predict(self,tgt):
        # output layer
        print("tgt:", tgt.shape)
        print(tgt)
        viirs_d = tgt[:, 0:20]
        common_d = tgt[:, 20:self.NUM]
        # dmval = self.ddm(tgt)
        dmval = self.ddm(viirs_d)
        combine_d = torch.cat((dmval, common_d), 1)
        tgt = self.feature(combine_d)
        tgt = self.central(tgt)
        tgt = self.fc(tgt)
        return tgt

# DDM model definition
class DDM(Module):
    # define model elements
    def __init__(self, n_inputs=20, n_outputs = 25):
        super(DDM, self).__init__()
        # # input to very beginning hidden layer
        self.hidden = Linear(n_inputs, 256)
        xavier_uniform_(self.hidden.weight)
        self.act = Sigmoid()
        # input to beginning hidden layer
        self.hidden0 = Linear(256, 256)
        xavier_uniform_(self.hidden0.weight)
        self.act0 = Sigmoid()
        # self.act0 = ReLU()
        # input to first hidden layer
        self.hidden1 = Linear(256, 256)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(256, 128)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # self.act2 = ReLU()
        # third hidden layer
        self.hidden3 = Linear(128, 64)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
        # self.act3 = ReLU()
        # 4th hidden layer
        self.hidden4 = Linear(64, n_outputs)
        xavier_uniform_(self.hidden4.weight)
        # self.act4 = Sigmoid()
        # # third hidden layer and output
        # self.hidden3 = Linear(64, 6)
        # xavier_uniform_(self.hidden3.weight)
        # self.act3 = Softmax(dim=1)
        self.dropout = Dropout(p=0.5)
        self.batchnorm = BatchNorm1d(256)
        self.batchnorm0 = BatchNorm1d(256)
        self.batchnorm1 = BatchNorm1d(256)
        self.batchnorm2 = BatchNorm1d(128)
        self.batchnorm3 = BatchNorm1d(64)
        self.batchnorm4 = BatchNorm1d(n_outputs)

    # forward propagate input
    def forward(self, X):
        # input to very first hidden layer
        X = self.hidden(X)
        X = self.batchnorm(X)
        X = self.act(X)
        X = self.dropout(X)
        # input to first hidden layer
        X = self.hidden0(X)
        X = self.batchnorm0(X)
        X = self.act0(X)
        X = self.dropout(X)
        # input to second hidden layer
        X = self.hidden1(X)
        X = self.batchnorm1(X)
        X = self.act1(X)
        X = self.dropout(X)
        # third hidden layer
        X = self.hidden2(X)
        X = self.batchnorm2(X)
        X = self.act2(X)
        X = self.dropout(X)
        # fourth hidden layer
        X = self.hidden3(X)
        X = self.batchnorm3(X)
        X = self.act3(X)
        X = self.dropout(X)
        # fifth hidden layer
        X = self.hidden4(X)
        X = self.batchnorm4(X)
        # X = self.act4(X)
        # X = self.dropout(X)
        # # output layer
        # X = self.hidden3(X)
        # X = self.act3(X)
        return X

# model definition n_inputs = 31
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs=25):
        super(MLP, self).__init__()
        # # input to very beginning hidden layer
        self.hidden = Linear(n_inputs, 128)
        kaiming_uniform_(self.hidden.weight, nonlinearity='relu')
        self.act = ReLU()
        # input to beginning hidden layer
        self.hidden0 = Linear(128, 256)
        kaiming_uniform_(self.hidden0.weight, nonlinearity='relu')
        self.act0 = ReLU()
        # input to first hidden layer
        self.hidden1 = Linear(256, 128)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(128, 64)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # # third hidden layer and output
        # self.hidden3 = Linear(64, 6)
        # xavier_uniform_(self.hidden3.weight)
        # self.act3 = Softmax(dim=1)
        self.dropout = Dropout(p=0.5)
        self.batchnorm = BatchNorm1d(128)
        self.batchnorm0 = BatchNorm1d(256)
        self.batchnorm1 = BatchNorm1d(128)
        self.batchnorm2 = BatchNorm1d(64)

    # forward propagate input
    def forward(self, X):
        # input to very first hidden layer
        X = self.hidden(X)
        X = self.batchnorm(X)
        X = self.act(X)
        X = self.dropout(X)
        # input to first hidden layer
        X = self.hidden0(X)
        X = self.batchnorm0(X)
        X = self.act0(X)
        X = self.dropout(X)
        # input to second hidden layer
        X = self.hidden1(X)
        X = self.batchnorm1(X)
        X = self.act1(X)
        X = self.dropout(X)
        # third hidden layer
        X = self.hidden2(X)
        X = self.batchnorm2(X)
        X = self.act2(X)
        # X = self.dropout(X)
        # # output layer
        # X = self.hidden3(X)
        # X = self.act3(X)
        return X

# classifer model definition n_inputs = 32 from the coral layer
class CLASSIFY(Module):
    # define model elements
    def __init__(self, n_inputs=32, n_outputs=6):
        super(CLASSIFY, self).__init__()
        # # input to very beginning hidden layer
        self.hidden = Linear(n_inputs, 64)
        kaiming_uniform_(self.hidden.weight, nonlinearity='relu')
        self.act = ReLU()
        # input to beginning hidden layer
        self.hidden0 = Linear(64, 128)
        kaiming_uniform_(self.hidden0.weight, nonlinearity='relu')
        self.act0 = ReLU()
        # input to first hidden layer
        self.hidden1 = Linear(128, 64)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(64, n_outputs)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        # self.act2 = ReLU()
        # # third hidden layer and output
        # self.hidden3 = Linear(64, 6)
        # xavier_uniform_(self.hidden3.weight)
        # self.act3 = Softmax(dim=1)
        self.dropout = Dropout(p=0.5)
        self.batchnorm = BatchNorm1d(64)
        self.batchnorm0 = BatchNorm1d(128)
        self.batchnorm1 = BatchNorm1d(64)
        self.batchnorm2 = BatchNorm1d(n_outputs)

    # forward propagate input
    def forward(self, X):
        # input to very first hidden layer
        X = self.hidden(X)
        X = self.batchnorm(X)
        X = self.act(X)
        X = self.dropout(X)
        # input to first hidden layer
        X = self.hidden0(X)
        X = self.batchnorm0(X)
        X = self.act0(X)
        X = self.dropout(X)
        # input to second hidden layer
        X = self.hidden1(X)
        X = self.batchnorm1(X)
        X = self.act1(X)
        X = self.dropout(X)
        # third hidden layer
        X = self.hidden2(X)
        X = self.batchnorm2(X)
        # X = self.act2(X)
        # X = self.dropout(X)
        # # output layer
        # X = self.hidden3(X)
        # X = self.act3(X)
        return X


def prepare_data(X_src, Y_src, X_tgt, Y_tgt, b_size=2048):
    # load the train dataset
    # src_train = CSVDataset(X_src, Y_src)
    # tgt_train = CSVDataset(X_tgt, Y_tgt)
    print("Y_src:")
    print(Y_src)
    print("Y_tgt:")
    print(Y_tgt)
    Y_src_1 = LabelEncoder().fit_transform(Y_src)
    Y_tgt_1 = LabelEncoder().fit_transform(Y_tgt)

    print("Y_src_1:")
    print(Y_src_1)
    print("Y_tgt_1:")
    print(Y_tgt_1)

    print("X_src:")
    print(X_src)
    print("X_tgt:")
    print(X_tgt)

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
    print("X_tgt:")
    print(X_tgt)

    # x_src = torch.from_numpy(X_src)
    # y_src = torch.from_numpy(Y_src_1)
    x_tgt = torch.from_numpy(X_tgt)
    # y_tgt = torch.from_numpy(Y_tgt_1)

    datasets = TensorDataset(x_tgt.float())
    # prepare data loaders
    train_dl = DataLoader(datasets, batch_size=b_size, shuffle=False)
    return train_dl

# train the model
def train_model(train_dat, valid_dat, model, n_epochs, lambda_, lambda_l2, device):
    aggre_losses = []
    aggre_losses_l2 = []
    aggre_losses_coral = []
    aggre_losses_classifier = []
    aggre_losses_classifier_tgt = []

    aggre_losses_valid = []
    aggre_losses_l2_valid = []
    aggre_losses_classifier_valid = []
    aggre_losses_classifier_valid_tgt = []
    aggre_losses_coral_valid = []

    aggre_train_acc = []
    aggre_valid_acc = []
    aggre_train_tgt_acc = []

    model.train()
    # define the optimization
    criterion = CrossEntropyLoss()
    l2loss = MSELoss()
    # optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
    # optimizer = torch.optim.SGD([{'params': model.feature.parameters()},
    #                              {'params':model.fc.parameters(),'lr':10*args.lr}],
    #                             lr= args.lr,momentum=args.momentum,weight_decay=args.weight_clay)

    # optimizer = RMSprop(model.parameters(), lr=0.0001)
    # optimizer1 = Adam([{'params': model.ddm.parameters(), 'lr': 0.01}], lr=0.01)
    # optimizer2 = RMSprop([{'params': model.feature.parameters()}, {'params': model.fc.parameters(), 'lr': 0.001}], lr=0.0001)
    optimizer = Adam([{'params': model.ddm.parameters(), 'lr': 0.001}, {'params': model.feature.parameters()}, {'params': model.fc.parameters(), 'lr': 0.000005}], lr=0.000005)

    es = EarlyStopping(patience=5)

    if torch.cuda.is_available():
      model = model.cuda()

    # enumerate epochs for DA
    j = 0
    for epoch in range(n_epochs):
        j += 1
        # enumerate mini batches of src domain and target domain
        train_steps = len(train_dat)
        print("DA train_steps:", train_steps)

        epoch_loss = 0
        epoch_loss_l2 = 0
        epoch_loss_classifier = 0
        epoch_loss_classifier_tgt = 0
        epoch_loss_coral = 0

        i = 0
        for it, (src_data, src_label, tgt_data, tgt_label) in enumerate(train_dat):
            # clear the gradients
            # optimizer1.zero_grad()
            # optimizer2.zero_grad()
            optimizer.zero_grad()
            if torch.cuda.is_available():
              tgt_data = tgt_data.to(device)
              src_data = src_data.to(device)
              src_label = src_label.to(device)
              tgt_label = tgt_label.to(device)

            # compute the model output
            # yhat = model(inputs)
            src_out, tgt_out, dm_out, centr1, centr2 = model(src_data, tgt_data)

            # calculate loss
            # loss = criterion(yhat, targets)
            # epoch_loss = loss
            loss_classifier = criterion(src_out, src_label)
            # print("src_label:")
            # print(src_label)
            loss_classifier_tgt = criterion(tgt_out, tgt_label)
            # equalRate(src_label.cpu(), tgt_label.cpu())
            # print("tgt_label:")
            # print(tgt_label)
            # loss_coral = CORAL(src_out, tgt_out)
            loss_coral = CORAL(centr1, centr2)

            loss_l2 = l2loss(dm_out, src_data[:, 0:25])
            sum_loss = lambda_ * loss_coral + loss_classifier + lambda_l2 * loss_l2 + loss_classifier_tgt * 0.5
            epoch_loss += sum_loss.item()
            epoch_loss_l2 += loss_l2.item()
            epoch_loss_classifier += loss_classifier.item()
            epoch_loss_classifier_tgt += loss_classifier_tgt.item()
            epoch_loss_coral += loss_coral.item()

            # credit assignment
            sum_loss.backward()
            # loss_l2.backward(retain_graph=True)
            # sum_loss.backward(retain_graph=True)
            # loss_l2.backward()
            # update model weights
            # optimizer1.step()
            # optimizer2.step()
            optimizer.step()
            i = i+1

        print('DA Train Epoch: {:2d} [{:2d}/{:2d}]\t'
              'Lambda: {:.4f}, Class: {:.6f}, CORAL: {:.6f}, l2_loss: {:.6f}, Total_Loss: {:.6f}'.format(
            epoch,
            i + 1,
            train_steps,
            lambda_,
            loss_classifier.item(),
            loss_coral.item(),
            loss_l2.item(),
            sum_loss.item()
        ))

        print('DA Train ith Epoch %d result:' % epoch)
        # calculate train src accuracy
        train_acc = evaluate_model_src(train_dat, model, device)
        aggre_train_acc.append(train_acc)
        print('DA train_acc: %.3f' % train_acc)

        # calculate train tgt accuracy
        train_tgt_acc = evaluate_model_tgt(train_dat, model, device)
        aggre_train_tgt_acc.append(train_tgt_acc)
        print('DA train_tgt_acc: %.3f' % train_tgt_acc)

        # # calculate valid accuracy
        valid_acc = evaluate_model_tgt(valid_dat, model, device)
        aggre_valid_acc.append(valid_acc)
        print('DA valid_tgt_acc: %.3f' % valid_acc)

        # # calculate test accuracy
        # test_acc = evaluate_model_tgt(test_dat, model, device)
        # aggre_test_acc.append(test_acc)
        # print('DA test_acc: %.3f' % test_acc)

        epoch_loss = epoch_loss / train_steps
        aggre_losses.append(epoch_loss)
        print(f'DA epoch: {j:3} sum loss: {epoch_loss:6.4f}')

        epoch_loss_l2 = epoch_loss_l2 / train_steps
        aggre_losses_l2.append(epoch_loss_l2)
        print(f'DA epoch: {j:3} l2 loss: {epoch_loss_l2:6.4f}')

        epoch_loss_classifier = epoch_loss_classifier / train_steps
        aggre_losses_classifier.append(epoch_loss_classifier)
        print(f'DA epoch: {j:3} classifier src loss: {epoch_loss_classifier:6.4f}')

        epoch_loss_classifier_tgt= epoch_loss_classifier_tgt / train_steps
        aggre_losses_classifier_tgt.append(epoch_loss_classifier_tgt)
        print(f'DA epoch: {j:3} classifier tgt loss: {epoch_loss_classifier_tgt:6.4f}')

        epoch_loss_coral = epoch_loss_coral / train_steps
        aggre_losses_coral.append(epoch_loss_coral)
        print(f'DA epoch: {j:3} coral loss: {epoch_loss_coral:6.4f}')

        # # calculate validate accuracy
        epoch_loss_valid, epoch_loss_l2_valid, epoch_loss_classifier_valid,  epoch_loss_coral_valid, epoch_loss_classifier_valid_tgt= evaluate_model_stop(valid_dat, model, lambda_, lambda_l2, device)  # evalution on dev set (i.e., holdout from training)
        aggre_losses_valid.append(epoch_loss_valid)
        aggre_losses_l2_valid.append(epoch_loss_l2_valid)
        aggre_losses_classifier_valid.append(epoch_loss_classifier_valid)
        aggre_losses_classifier_valid_tgt.append(epoch_loss_classifier_valid_tgt)
        aggre_losses_coral_valid.append(epoch_loss_coral_valid)

        print(f'DA epoch: {j:3} valid sum loss: {epoch_loss_valid:6.4f}')
        print(f'DA epoch: {j:3} valid l2 loss: {epoch_loss_l2_valid:6.4f}')
        print(f'DA epoch: {j:3} valid classifier loss: {epoch_loss_classifier_valid:6.4f}')
        print(f'DA epoch: {j:3} valid tgt classifier loss: {epoch_loss_classifier_valid_tgt:6.4f}')
        print(f'DA epoch: {j:3} valid coral loss: {epoch_loss_coral_valid:6.4f}')

        if es.step(np.array(epoch_loss_classifier_valid_tgt)):
            print(f'Early Stopping Criteria Met!')
            break  # early stop criterion is met, we can stop now

# evaluate the validation loss for early stop
def evaluate_model_stop(valid_dl, model, lambda_, lambda_l2, device):
  model.eval()
  # define the optimization
  criterion = CrossEntropyLoss()
  l2loss = MSELoss()
  epoch_loss = 0
  epoch_loss_l2 = 0
  epoch_loss_classifier = 0
  epoch_loss_classifier_valid = 0
  epoch_loss_coral = 0

  valid_steps = len(valid_dl)
  print("DA train_steps:", valid_steps)

  for i, (src_data, src_label, tgt_data, tgt_label) in enumerate(valid_dl):
    if torch.cuda.is_available():
      tgt_data = tgt_data.to(device)
      src_data = src_data.to(device)
      src_label = src_label.to(device)
      tgt_label = tgt_label.to(device)

    with torch.no_grad():
      src_out, tgt_out, dm_out, centr1, centr2 = model(src_data, tgt_data)
    # calculate loss
    loss_classifier = criterion(src_out, src_label)
    loss_classifier_valid = criterion(tgt_out, tgt_label)
    # loss_coral = CORAL(src_out, tgt_out)
    loss_coral = CORAL(centr1, centr2)
    loss_l2 = l2loss(dm_out, src_data[:, 0:25])
    sum_loss = lambda_ * loss_coral + loss_classifier + lambda_l2 * loss_l2 + loss_classifier_valid * 0.5
    epoch_loss += sum_loss.item()
    epoch_loss_l2 += loss_l2.item()
    epoch_loss_classifier += loss_classifier.item()
    epoch_loss_classifier_valid += loss_classifier_valid.item()
    epoch_loss_coral += loss_coral.item()
  
  return epoch_loss / valid_steps, epoch_loss_l2 / valid_steps, epoch_loss_classifier / valid_steps, epoch_loss_coral / valid_steps, epoch_loss_classifier_valid / valid_steps

# evaluate the model source
def evaluate_model_src(test_dl, model, device):
    model.eval()
    predictions, actuals = list(), list()
    NUM = 22
    # test_steps = len(test_dl)
    # iter_test = iter(test_dl)
    for i, (src_data, src_label, tgt_data, tgt_label) in enumerate(test_dl):
    # for i in range(test_steps):
        # evaluate the model on the test set
        # tgt_data, targets = iter_test.next()
        if torch.cuda.is_available():
          src_data = src_data.to(device)
          src_label = src_label.to(device)

        tgt_data = src_data
        targets = src_label
        # temp = torch.zeros((tgt_data.shape[0], DIFFERECE_COL))
        # tmp_data = torch.cat((tgt_data, temp), 1)
        with torch.no_grad():
          yhat, _, _, _, _ = model(tgt_data, tgt_data[:,0:NUM])
        # retrieve numpy array
        yhat = yhat.detach().cpu().numpy()
        actual = targets.cpu().numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# evaluate the model target
# TODO: improve the predict function 
def evaluate_model_tgt(test_dl, model, device):
    model.eval()
    predictions, actuals = list(), list()
    # test_steps = len(test_dl)
    # iter_test = iter(test_dl)
    for i, (src_data, src_label, target_data, target_label) in enumerate(test_dl):
    # for i in range(test_steps):
        # evaluate the model on the test set
        if torch.cuda.is_available():
          target_data = target_data.to(device)
          target_label = target_label.to(device)

        tgt_data = target_data
        targets = target_label
        DIFFERECE_COL = 5
        temp = torch.zeros((tgt_data.shape[0], DIFFERECE_COL))
        temp = temp.to(device)
        tmp_data = torch.cat((tgt_data, temp), 1)
        # temp = torch.zeros((tgt_data.shape[0], DIFFERECE_COL))
        # tmp_data = torch.cat((tgt_data, temp), 1)
        with torch.no_grad():
          _, yhat, _, _, _ = model(tmp_data, tgt_data)
        # retrieve numpy array
        yhat = yhat.detach().cpu().numpy()
        actual = targets.cpu().numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

def test_all(prefix, filenames, b_size, device):
  for i in range(len(filenames)):
    X_s_test, Y_s_test, X_t_test, Y_t_test = load_test_data(prefix, filenames[i])

    # X_t = x_train_pt
    # Y_t = y_train

    # # X_s_test = X_s_test
    # # Y_s_test = Y_s_test
    # X_t_test = x_test_pt_test
    # Y_t_test = Y_s_test


    test_dat = prepare_data(X_s_test, Y_s_test, X_t_test, Y_t_test, b_size)

    # train_tgt, test_tgt = prepare_data(X_t, Y_t, X_t_test, Y_t_test)


    # evaluate the model
    acc = evaluate_model_tgt(test_dat, model, device)
    all_results.append(acc)
    print(filenames[i])
    print('test Accuracy: %.3f' % acc)


def preProcessing(training_data_path, model_saving_path, b_size):
  # load the training data
  # train_file = 'train10.npz'
  # prefix = '/content/content/My Drive/Colab Notebooks/kddworkshop/weak/'
  prefix = training_data_path
  # train_file = '2013_mon1.npz'
  #train_file = '2013.npz'
  #train_file_1 = '2013.npz'
  #train_file_2 = '2014.npz'
  #train_file_3 = '2015.npz'
  #train_file_4 = '2016.npz'
  file_count = 0
  train_files = glob.glob(prefix + '/*.npz')
  for train_file in train_files: 
    if file_count < 1:
      X_v, X_c, Y_v, Y_c  = loadData(train_file)
    else: 
       X_v_1, X_c_1, Y_v_1, Y_c_1 = loadData(train_file)
       X_v = np.concatenate((X_v, X_v_1), axis = 0)
       X_c = np.concatenate((X_c, X_c_1), axis = 0)
       Y_v = np.concatenate((Y_v, Y_v_1), axis=0)
       Y_c = np.concatenate((Y_c, Y_c_1), axis=0)
       del X_v_1, X_c_1, Y_v_1, Y_c_1
  #train_file_1 = '2017_jan_day_005.npz'
  #train_file_2 = '2017_jan_day_019.npz'
  #train_file_3 = '2017_jan_day_024.npz'
  #train_file_4 = '2017_jan_day_030.npz'

  # train_file = '4yr_jan.npz'
  # X_v, X_c, Y_v, Y_c = loadData(prefix, train_file)
  #X_v_1, X_c_1, Y_v_1, Y_c_1 = loadData(prefix, train_file_1)
  #X_v_2, X_c_2, Y_v_2, Y_c_2 = loadData(prefix, train_file_2)
  #X_v_3, X_c_3, Y_v_3, Y_c_3 = loadData(prefix, train_file_3)
  #X_v_4, X_c_4, Y_v_4, Y_c_4 = loadData(prefix, train_file_4)
  #X_v_1, X_c_1, Y_v_1, Y_c_1 = loadData(prefix, train_file_1)
  #X_v = X_v_1
  #X_c = X_c_1
  #Y_v = Y_v_1
  #Y_c = Y_c_1

  #X_v_1, X_c_1, Y_v_1, Y_c_1 = loadData(prefix, train_file_2)
  #X_v = np.concatenate((X_v, X_v_1), axis = 0)
  #X_c = np.concatenate((X_c, X_c_1), axis = 0)
  #Y_v = np.concatenate((Y_v, Y_v_1), axis=0)
  #Y_c = np.concatenate((Y_c, Y_c_1), axis=0)

  #X_v_1, X_c_1, Y_v_1, Y_c_1 = loadData(prefix, train_file_3)
  #X_v = np.concatenate((X_v, X_v_1), axis = 0)
  #X_c = np.concatenate((X_c, X_c_1), axis = 0)
  #Y_v = np.concatenate((Y_v, Y_v_1), axis=0)
  #Y_c = np.concatenate((Y_c, Y_c_1), axis=0)

  #X_v_1, X_c_1, Y_v_1, Y_c_1 = loadData(prefix, train_file_4)
  #X_v = np.concatenate((X_v, X_v_1), axis = 0)
  #X_c = np.concatenate((X_c, X_c_1), axis = 0)
  #Y_v = np.concatenate((Y_v, Y_v_1), axis=0)
  #Y_c = np.concatenate((Y_c, Y_c_1), axis=0)

  #X_v = np.concatenate((X_v_1, X_v_2, X_v_3, X_v_4), axis=0)
  #X_c = np.concatenate((X_c_1, X_c_2, X_c_3, X_c_4), axis=0)
  #Y_v = np.concatenate((Y_v_1, Y_v_2, Y_v_3, Y_v_4), axis=0)
  #Y_c = np.concatenate((Y_c_1, Y_c_2, Y_c_3, Y_c_4), axis=0)


  Y = np.concatenate((Y_v, Y_c), axis=1)
  print ("X_v.shape:", X_v.shape)
  #print (Y_v.shape)
  print ("X_c.shape:", X_c.shape)
  #print (Y_c.shape)
  print ("Y.shape:", Y.shape)

  # combine data and split latter to define ground truth for MLR
  # from sklearn.linear_model import LinearRegression
  n1=20
  n2=25
  X=np.concatenate((X_v, X_c), axis=1)

  # Y=Y_v
  print (X.shape)
  # print (Y_v)
  x_train, x_valid, y_train, y_valid = train_test_split(X, Y,
                                                      test_size=0.3,
                                                      random_state=0,
                                                      stratify=Y)

  del X
  del Y
  del X_v
  del X_c
  del Y_v, Y_c

  # Use index shuffling to avoid out of memory issue 
  #x_ids = list(range(len(X)))
  #x_train_ids, x_valid_ids, y_train, y_valid = train_test_split(x_ids, Y, test_size = 0.3, random_state=0, stratify=Y)
  #x_train = X[x_train_ids]
  #x_valid = X[x_valid_ids]

  # feature scaling
  sc_X = StandardScaler()
  x_train=sc_X.fit_transform(x_train)
  x_valid=sc_X.transform(x_valid)
  # x_test=sc_X.fit_transform(x_test)

  # save the scaler
  dump(sc_X, open(model_saving_path + '/scaler.pkl', 'wb'))
  print("saved scaler done")

  x_train_v = x_train[:, 0:20]
  x_train_c = x_train[:, 20:45]
  x_train_comm = x_train[:, 45:47]
  x_train_src = x_train[:, 20:47]
  y_train_src = np.delete(y_train, 0, 1)
  y_train_tgt = np.delete(y_train, 1, 1)

  print(x_train_v.shape)
  print(x_train_c.shape)
  print(x_train_comm.shape)
  print(x_train_src.shape)
  print(y_train.shape)
  print(y_train_src.shape)
  print(y_train_tgt.shape)

  x_valid_src = x_valid[:, 20:51]
  x_valid_v = x_valid[:, 0:20]
  x_valid_c = x_valid[:, 20:45]
  x_valid_comm = x_valid[:, 45:47]
  y_valid_src = np.delete(y_valid, 0, 1)
  y_valid_tgt = np.delete(y_valid, 1, 1)

  print(x_valid_v.shape)
  print(x_valid_c.shape)
  print(x_valid_comm.shape)
  print(x_valid_src.shape)
  print(y_valid.shape)
  print(y_valid_src.shape)
  print(y_valid_tgt.shape)

  x_train_c_pt = x_train_v
  x_valid_c_pt = x_valid_v

  print(x_train_c_pt.shape)
  print(x_valid_c_pt.shape)

  # DLR imputed target domain
  x_train_pt = np.concatenate((x_train_c_pt, x_train_comm),axis=1)
  print(x_train_pt.shape)

  x_valid_pt = np.concatenate((x_valid_c_pt, x_valid_comm),axis=1)
  print(x_valid_pt.shape)

  #X_s_test, Y_s_test, x_test_pt_test, Y_t_test = load_test_data(prefix, '2017_jan_day_005.npz')

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

  # # test data
  # X_s_test = X_s_test
  # Y_s_test = Y_s_test
  # X_t_test = x_test_pt_test
  # Y_t_test = Y_t_test


  train_dat = prepare_data(X_s, Y_s, X_t, Y_t, b_size)
  valid_dat = prepare_data(X_s_valid, Y_s_valid, X_t_valid, Y_t_valid, b_size)
  #test_dat = prepare_data(X_s_test, Y_s_test, X_t_test, Y_t_test)

  return train_dat, valid_dat


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--training_data_path")
  parser.add_argument("--model_saving_path")
  args = parser.parse_args()

  NUM_LALBELS = 3
  EPOCHS = 10
  lambda_ = 0.001
  lambda_l2 = 0.05
  BATCH_SIZE = 2048

  _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("torch.cuda.is_available:", torch.cuda.is_available())

  #load the training data
  train_dat, valid_dat = preProcessing(args.training_data_path, args.model_saving_path, b_size=BATCH_SIZE)

  # initiate the model
  model = Deep_coral(num_classes=NUM_LALBELS)
  # train the model
  train_model(train_dat, valid_dat, model, EPOCHS, lambda_, lambda_l2, _device)

  #evaluate the model on valid data
  acc = evaluate_model_tgt(valid_dat, model, _device)
  print('Accuracy: %.3f' % acc)

  # save the model
  torch.save(model, args.model_saving_path + 'model.pth')




