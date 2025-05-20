#ABI_modelVCMv2.py
import torch
from torch.nn import Module, Linear, ReLU, Dropout, BatchNorm1d, Sigmoid
from torch.nn.init import xavier_uniform_

# Define feature sizes
LATENT_DIM = 8
DDM_NUM = 16  # ABI features
NUM = 22  # ABI (16) + Angles (4) + Latlon (2)
DIFFERECE_COL = 5  # Additional columns for domain adaptation
COMMON_FEATURES = 6  # Shared features

class Deep_coral(Module):
    def __init__(self, num_classes=2):
        super(Deep_coral, self).__init__()

        # Domain discrepancy model (DDM) - Transforms ABI features
        self.ddm = DDM(n_inputs=DDM_NUM, n_outputs=DDM_NUM + DIFFERECE_COL)

        # Feature extraction model (MLP expects 27 features)
        self.feature = MLP(n_inputs=NUM + DIFFERECE_COL)

        #  FIX: Change central layer to expect 32 features instead of 64
        self.central = Linear(32, 32)  #  Fix applied here
        xavier_uniform_(self.central.weight)

        # Classification model
        self.fc = CLASSIFY(32, num_classes)

    def forward(self, src, tgt):
        """
        Forward pass through DeepCORAL model for domain adaptation.
        - `src` = Source domain (ABI)
        - `tgt` = Target domain (CALIPSO)
        """

        #  Transform `src_data` properly before passing to MLP
        src_transformed = self.ddm(src)  # Converts 16 → 21 (DDM_NUM + DIFFERECE_COL)
        common_features_src = tgt[:, 16:NUM]  # Extract common features (6)
        src_combined = torch.cat((src_transformed, common_features_src), dim=1)  # 21 + 6 = 27

        #  Pass correctly transformed `src` to `MLP`
        src = self.feature(src_combined)
        centr1 = self.central(src)  #  This now expects (batch_size, 32)
        src = self.fc(centr1)

        #  Process target (CALIPSO)
        abi_features = tgt[:, :16]  # Extract ABI features from target
        common_features_tgt = tgt[:, 16:NUM]  # Extract common features (6)

        # Transform ABI features using DDM
        dmval = self.ddm(abi_features)
        combine_d = torch.cat((dmval, common_features_tgt), dim=1)  # 21 + 6 = 27

        #  Ensure `tgt` also has 27 features
        tgt = self.feature(combine_d)
        centr2 = self.central(tgt)  #  This now expects (batch_size, 32)
        tgt = self.fc(centr2)

        return src, tgt, dmval, centr1, centr2



class DDM(Module):
    """
    Domain Discrepancy Model (DDM) - Transforms ABI features into a common feature space.
    """
    def __init__(self, n_inputs=16, n_outputs=21):
        super(DDM, self).__init__()

        # First hidden layer (Input → 256)
        self.hidden = Linear(n_inputs, 256)
        xavier_uniform_(self.hidden.weight)
        self.batchnorm = BatchNorm1d(256)
        self.act = Sigmoid()
        self.dropout = Dropout(0.5)

        # Second hidden layer (256 → 256)
        self.hidden0 = Linear(256, 256)
        xavier_uniform_(self.hidden0.weight)
        self.batchnorm0 = BatchNorm1d(256)
        self.act0 = Sigmoid()
        self.dropout0 = Dropout(0.5)

        # Third hidden layer (256 → 256)
        self.hidden1 = Linear(256, 256)
        xavier_uniform_(self.hidden1.weight)
        self.batchnorm1 = BatchNorm1d(256)
        self.act1 = Sigmoid()
        self.dropout1 = Dropout(0.5)

        # Fourth hidden layer (256 → 128)
        self.hidden2 = Linear(256, 128)
        xavier_uniform_(self.hidden2.weight)
        self.batchnorm2 = BatchNorm1d(128)
        self.act2 = Sigmoid()
        self.dropout2 = Dropout(0.5)

        # Fifth hidden layer (128 → 64)
        self.hidden3 = Linear(128, 64)
        xavier_uniform_(self.hidden3.weight)
        self.batchnorm3 = BatchNorm1d(64)
        self.act3 = Sigmoid()
        self.dropout3 = Dropout(0.5)

        # Output layer (64 → n_outputs)
        self.hidden4 = Linear(64, n_outputs)
        xavier_uniform_(self.hidden4.weight)

        self.dropout = Dropout(p=0.5)
        self.batchnorm = BatchNorm1d(256)
        self.batchnorm0 = BatchNorm1d(256)
        self.batchnorm1 = BatchNorm1d(256)
        self.batchnorm2 = BatchNorm1d(128)
        self.batchnorm3 = BatchNorm1d(64)
        self.batchnorm4 = BatchNorm1d(n_outputs)

    def forward(self, X):
        """
        Forward pass through the DDM model.
        """

        # First hidden layer
        X = self.hidden(X)
        X = self.batchnorm(X)
        X = self.act(X)
        X = self.dropout(X)

        # Second hidden layer
        X = self.hidden0(X)
        X = self.batchnorm0(X)
        X = self.act0(X)
        X = self.dropout0(X)

        # Third hidden layer
        X = self.hidden1(X)
        X = self.batchnorm1(X)
        X = self.act1(X)
        X = self.dropout1(X)

        # Fourth hidden layer
        X = self.hidden2(X)
        X = self.batchnorm2(X)
        X = self.act2(X)
        X = self.dropout2(X)

        # Fifth hidden layer
        X = self.hidden3(X)
        X = self.batchnorm3(X)
        X = self.act3(X)
        X = self.dropout3(X)

        # Output layer
        X = self.hidden4(X)
        X = self.batchnorm4(X)

        return X

class MLP(Module):
    """
    Feature extractor using a Multi-Layer Perceptron (MLP).
    """
    def __init__(self, n_inputs=27):  #  **Fix: Ensure MLP expects 27 inputs**
        super(MLP, self).__init__()

        self.hidden1 = Linear(n_inputs, 128)
        xavier_uniform_(self.hidden1.weight)
        self.batchnorm1 = BatchNorm1d(128)
        self.act1 = ReLU()
        self.dropout1 = Dropout(0.3)

        self.hidden2 = Linear(128, 64)
        xavier_uniform_(self.hidden2.weight)
        self.batchnorm2 = BatchNorm1d(64)
        self.act2 = ReLU()
        self.dropout2 = Dropout(0.3)

        self.hidden3 = Linear(64, 32)
        xavier_uniform_(self.hidden3.weight)

    def forward(self, X):
        """
        Forward pass through MLP feature extractor.
        """
        X = self.hidden1(X)
        X = self.batchnorm1(X)
        X = self.act1(X)
        X = self.dropout1(X)

        X = self.hidden2(X)
        X = self.batchnorm2(X)
        X = self.act2(X)
        X = self.dropout2(X)

        X = self.hidden3(X)
        return X


class CLASSIFY(Module):
    """
    Classification model for final predictions.
    """
    def __init__(self, n_inputs=32, n_outputs=2):
        super(CLASSIFY, self).__init__()

        self.hidden1 = Linear(n_inputs, 64)
        xavier_uniform_(self.hidden1.weight)
        self.batchnorm1 = BatchNorm1d(64)
        self.act1 = ReLU()
        self.dropout1 = Dropout(0.3)

        self.hidden2 = Linear(64, 128)
        xavier_uniform_(self.hidden2.weight)
        self.batchnorm2 = BatchNorm1d(128)
        self.act2 = ReLU()
        self.dropout2 = Dropout(0.3)

        self.hidden3 = Linear(128, 64)
        xavier_uniform_(self.hidden3.weight)
        self.batchnorm3 = BatchNorm1d(64)
        self.act3 = ReLU()
        self.dropout3 = Dropout(0.3)

        self.output = Linear(64, n_outputs)
        xavier_uniform_(self.output.weight)

    def forward(self, X):
        """
        Forward pass through classification model.
        """
        X = self.hidden1(X)
        X = self.batchnorm1(X)
        X = self.act1(X)
        X = self.dropout1(X)

        X = self.hidden2(X)
        X = self.batchnorm2(X)
        X = self.act2(X)
        X = self.dropout2(X)

        X = self.hidden3(X)
        X = self.batchnorm3(X)
        X = self.act3(X)
        X = self.dropout3(X)

        X = self.output(X)
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
NUM_LALBELS = 2
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

