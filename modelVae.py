from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Softmax
from torch.nn import Sequential
from torch.nn import LeakyReLU
from torch.nn import Tanh
from torch.nn import Module
from torch.nn import Dropout
from torch.nn import BatchNorm1d
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import torch

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prefix = '/Users/nizhao/xin/access/data/weak/saved_model/'

lambda_ = 0.001
lambda_l2 = 0.05
# NUM = 31
DDM_NUM = 20
# NUM = 26
NUM = 22
DIFFERECE_COL = 5
# COMMON_FEATURES = 6
COMMON_FEATURES = 2
LATENT_DIM = 512
BATCH_SIZE = 256
NUM_LALBELS = 3


# DDM model definition
class DDM(Module):
    # define model elements
    # -4, remove 'VIIRS_SZA','VIIRS_SAA','VIIRS_VZA','VIIRS_VAA'
    def __init__(self, n_inputs=DDM_NUM-4, n_outputs = DDM_NUM+DIFFERECE_COL):
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
    def __init__(self, n_inputs=DDM_NUM+DIFFERECE_COL):
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


# Define VAE based heteregenous domain adaptation
# DDM_NUM+DIFFERECE_COL+COMMON_FEATURES
class Deep_VAE(Module):
    def __init__(self, num_classes=NUM_LALBELS):
        super(Deep_VAE, self).__init__()
        # NUM:26 -> 22
        self.src_vae = VAE_DA(n_inputs=DDM_NUM + DIFFERECE_COL + COMMON_FEATURES)
        # self.tgt_vae = VAE_DA(n_inputs = DDM_NUM+COMMON_FEATURES)
        self.tgt_vae = VAE_DA(n_inputs=DDM_NUM + DIFFERECE_COL + COMMON_FEATURES)
        self.ddm = DDM(n_inputs=DDM_NUM, n_outputs=DDM_NUM + DIFFERECE_COL)
        # self.feature = MLP(n_inputs=NUM+DIFFERECE_COL)
        self.feature = MLP(n_inputs=LATENT_DIM)
        self.central = Linear(64, 32)  # correlation layer
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

    def forward(self, src, tgt):
        decoded_src, mu_src, logvar_src, z_src = self.src_vae(src)
        src = self.feature(mu_src)
        # src = self.feature(src)
        centr1 = self.central(src)
        src = self.fc(centr1)
        # output layer
        viirs_d = tgt[:, 4:20]  # bandM1-M16, remove VZA,VAA,SZA,SAA
        common_d = tgt[:, 20:NUM]  # NUM = 26>22
        dmval = self.ddm(viirs_d)
        combine_d = torch.cat((dmval, common_d), 1)
        # decoded_tgt, mu_tgt, logvar_tgt = self.tgt_vae(tgt)
        decoded_tgt, mu_tgt, logvar_tgt, z_tgt = self.tgt_vae(combine_d)
        tgt = self.feature(mu_tgt)
        centr2 = self.central(tgt)
        # tgt = self.feature(tgt)
        tgt = self.fc(centr2)
        return src, tgt, dmval, centr1, centr2, decoded_src, mu_src, logvar_src, decoded_tgt, mu_tgt, logvar_tgt, combine_d, z_src, z_tgt

    def pretrain(self, tgt):
        # output layer
        viirs_d = tgt[:, 4:20]
        common_d = tgt[:, 20:NUM]
        # dmval = self.ddm(tgt)
        dmval = self.ddm(viirs_d)
        # combine_d = torch.cat((dmval, common_d), 1)
        # tgt = self.feature(combine_d)
        # tgt = self.fc(tgt)
        return dmval


class VAE_DA(Module):
    def __init__(self, n_inputs):
        super().__init__()

        # self.fc = CLASSIFY(n_outputs, num_classes)
        # self.LeakyReLU = nn.LeakyReLU(0.2)
        # self.fc = CLASSIFY(n_inputs, num_classes)
        # self.dropout = Dropout(p=0.25)

        # self.fc = CLASSIFY(d, num_classes)
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
            # Sigmoid(),
            # Dropout(p=0.5),
            # Linear(d ** 2, d * 2)
        )

        self.decoder = Sequential(
            Linear(LATENT_DIM, 128),
            BatchNorm1d(128),
            LeakyReLU(0.2),
            Linear(128, 256),
            BatchNorm1d(256),
            LeakyReLU(0.2),
            Linear(256, n_inputs),
            # BatchNorm1d(n_outputs),
            # LeakyReLU(0.2),

            # Dropout(p=0.5),
            # Linear(d ** 2, 784),
            # Sigmoid(),
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu_logvar = self.encoder(x).view(-1, 2, LATENT_DIM)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        decoded_val = self.decoder(z)
        # tgt_va = self.fc(mu)
        return decoded_val, mu, logvar, z



# Define VAE based heteregenous domain adaptation
# DDM_NUM+DIFFERECE_COL+COMMON_FEATURES
class Deep_Dual_VAE(Module):
    def __init__(self,num_classes = NUM_LALBELS):
        super(Deep_Dual_VAE,self).__init__()
        #NUM:26
        # self.src_vae = VAE_DA(n_inputs = DDM_NUM+DIFFERECE_COL+COMMON_FEATURES)
        # self.tgt_vae = VAE_DA(n_inputs = DDM_NUM+DIFFERECE_COL+COMMON_FEATURES)
        self.src_vae = VAE_DA(n_inputs = DDM_NUM+DIFFERECE_COL)
        self.tgt_vae = VAE_DA(n_inputs = DDM_NUM+DIFFERECE_COL)
        # self.src_vae.load_state_dict(torch.load(prefix + '/model_vae.pt'))
        self.tgt_vae.load_state_dict(torch.load(prefix + '/model_vae_best.pt', map_location=torch.device('cpu')))
        # self.src_vae.to(_device)
        # self.tgt_vae.to(_device)
        self.ddm = DDM(n_inputs=DDM_NUM-4,n_outputs=DDM_NUM+DIFFERECE_COL)
        # self.ddm.load_state_dict(torch.load(prefix + '/model_ddm.pt'))
        # self.feature = MLP(n_inputs=NUM+DIFFERECE_COL)
        # self.feature = MLP(n_inputs=LATENT_DIM)
        self.feature = MLP(n_inputs=LATENT_DIM + COMMON_FEATURES)
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
        calipso_d = src[:, 0:DDM_NUM+DIFFERECE_COL]
        common_src_d = src[:, DDM_NUM+DIFFERECE_COL:DDM_NUM+DIFFERECE_COL + COMMON_FEATURES]
        # decoded_src, mu_src, logvar_src, z_src = self.src_vae(src)
        decoded_src, mu_src, logvar_src, z_src = self.src_vae(calipso_d)
        combine_src = torch.cat((mu_src, common_src_d), 1)
        src = self.feature(combine_src)
        # src = self.feature(mu_src)
        centr1 = self.central(src)
        src = self.fc(centr1)
        # output layer DDM_NUM= 20
        viirs_d = tgt[:, 4:DDM_NUM]
        common_d = tgt[:, DDM_NUM:DDM_NUM+COMMON_FEATURES]
        dmval = self.ddm(viirs_d)
        combine_d = torch.cat((dmval, common_d), 1) # this is returned in the function TODO check it
        decoded_tgt, mu_tgt, logvar_tgt, z_tgt = self.tgt_vae(dmval)
        combine_tgt = torch.cat((mu_tgt, common_src_d), 1)
        tgt = self.feature(combine_tgt)
        # tgt = self.feature(mu_tgt)
        centr2 = self.central(tgt)
        tgt = self.fc(centr2)
        return src,tgt,dmval,centr1,centr2,decoded_src, mu_src, logvar_src,decoded_tgt, mu_tgt, logvar_tgt, combine_d, z_src, z_tgt

    def pretrain(self,tgt):
        # output layer
        viirs_d = tgt[:, 4:20]
        common_d = tgt[:, 20:22]
        # dmval = self.ddm(tgt)
        dmval = self.ddm(viirs_d)
        # combine_d = torch.cat((dmval, common_d), 1)
        # tgt = self.feature(combine_d)
        # tgt = self.fc(tgt)
        return dmval

    def predict(self,tgt):
        # output layer
        viirs_d = tgt[:, 4:20]
        common_d = tgt[:, DDM_NUM:DDM_NUM+COMMON_FEATURES]
        # dmval = self.ddm(tgt)
        dmval = self.ddm(viirs_d)
        # combine_d = torch.cat((dmval, common_d), 1)

        decoded_tgt, mu_tgt, logvar_tgt, z_tgt = self.tgt_vae(dmval)
        combine_tgt = torch.cat((mu_tgt, common_d), 1)
        tgt = self.feature(combine_tgt)
        centr2 = self.central(tgt)
        tgt = self.fc(centr2)

        # tgt = self.feature(combine_d)
        # tgt = self.central(tgt)
        # tgt = self.fc(tgt)
        return tgt