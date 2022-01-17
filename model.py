"""
Cloud phase prediction 
01/01/2021

@@author: Xin Huang
"""

from numpy import vstack
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import glob
from pickle import dump
import pandas as pd


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
        viirs_d = tgt[:, 0:20]
        common_d = tgt[:, 20:self.NUM]
        # dmval = self.ddm(tgt)
        dmval = self.ddm(viirs_d)
        return dmval

    def predict(self,tgt):
        # output layer
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

        return X



