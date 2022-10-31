"""
Cloud phase prediction 
01/01/2021

@@author: Xin Huang
"""

from torch import flatten
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Softmax
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import LeakyReLU
from torch.nn import Tanh
from torch.nn import Dropout
from torch.nn import BatchNorm1d
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import torch

LATENT_DIM = 8
DDM_NUM = 20
NUM = 26
DIFFERECE_COL = 5
COMMON_FEATURES = 6

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

class VAE_DA(Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.conv1 = Conv1d(n_inputs, 36, 3, stride=1)
        self.conv2 = Conv1d(36, n_inputs, 3, stride=1)
        self.encoder = Sequential(
            Linear(n_inputs, 256),
            BatchNorm1d(256),
            LeakyReLU(0.2),
            Linear(256, 128),
            BatchNorm1d(128),
            LeakyReLU(0.2),
            Linear(128, LATENT_DIM * 2),
            BatchNorm1d(LATENT_DIM * 2),
            LeakyReLU(0.2),
            Tanh(),
        )

        self.decoder = Sequential(
            Linear(LATENT_DIM, 128),
            BatchNorm1d(128),
            LeakyReLU(0.2),
            Linear(128, 256),
            BatchNorm1d(256),
            LeakyReLU(0.2),
            Linear(256, n_inputs),
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        x = x.reshape(int(x.shape[0] / 5), 5, x.shape[1])
        # conv1d takes (N, C_in, L_in) and outputs (N, C_out, L_out),
        # C_in = n_inputs= x.shape[1], L_in = 5; C_out=user set = 36; L_out = 1
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = flatten(x, start_dim=1)
        mu_logvar = self.encoder(x).view(-1, 2, LATENT_DIM)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        decoded_val = self.decoder(z)
        # tgt_va = self.fc(mu)
        return decoded_val, mu, logvar, z


# Define VAE based heteregenous domain adaptation
# DDM_NUM+DIFFERECE_COL+COMMON_FEATURES
NUM_LALBELS = 3
class Deep_VAE(Module):
    def __init__(self,num_classes = NUM_LALBELS):
        super(Deep_VAE,self).__init__()
        #NUM:26
        self.src_vae = VAE_DA(n_inputs = DDM_NUM+DIFFERECE_COL+COMMON_FEATURES)
        #self.tgt_vae = VAE_DA(n_inputs = DDM_NUM+COMMON_FEATURES)
        self.tgt_vae = VAE_DA(n_inputs = DDM_NUM+DIFFERECE_COL+COMMON_FEATURES)
        self.ddm = DDM(n_inputs=DDM_NUM,n_outputs=DDM_NUM+DIFFERECE_COL)
        # self.feature = MLP(n_inputs=NUM+DIFFERECE_COL)
        self.feature = MLP(n_inputs=LATENT_DIM)
        self.central = Linear(64,32) # correlation layer
        xavier_uniform_(self.central.weight)

        self.fc = CLASSIFY(32, num_classes)

    def forward(self,src,tgt):
        decoded_src, mu_src, logvar_src, z_src = self.src_vae(src)
        src = self.feature(mu_src)
        # src = self.feature(src)
        centr1 = self.central(src)
        src = self.fc(centr1)
        # output layer
        viirs_d = tgt[:, 0:20]
        common_d = tgt[:, 20:26]
        dmval = self.ddm(viirs_d)
        combine_d = torch.cat((dmval, common_d), 1)
        # decoded_tgt, mu_tgt, logvar_tgt = self.tgt_vae(tgt)
        decoded_tgt, mu_tgt, logvar_tgt, z_tgt = self.tgt_vae(combine_d)
        tgt = self.feature(mu_tgt)
        centr2 = self.central(tgt)
        # tgt = self.feature(tgt)
        tgt = self.fc(centr2)
        return src,tgt,dmval,centr1,centr2,decoded_src, mu_src, logvar_src,decoded_tgt, mu_tgt, logvar_tgt, combine_d, z_src, z_tgt

    def pretrain(self,tgt):
        # output layer
        viirs_d = tgt[:, 0:20]
        common_d = tgt[:, 20:26]
        # dmval = self.ddm(tgt)
        dmval = self.ddm(viirs_d)
        return dmval
