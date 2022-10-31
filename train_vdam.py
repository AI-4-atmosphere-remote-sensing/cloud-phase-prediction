import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from numpy import vstack
from numpy import argmax
from torch.optim import SGD, RMSprop, Adam
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from data_utils_vdam import preProcessing
from model import Deep_VAE

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='linear', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd(self, X, Y):
        delta = X.mean(axis=0) - Y.mean(axis=0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            return loss

def setWeight(a, b):
  c = torch.ones([a.shape[0]], dtype=torch.float32)
  # print("c:", c.shape)
  # print(c)
  for i in range(a.shape[0]):
    if a[i] == b[i]:
      c[i] = 1.5
    elif a[i] == 3:
      c[i] = 1.25
  # print("weight c:", c.shape)
  # print(c)
  return c

# train the model
aggre_losses = []
aggre_losses_l2 = []
aggre_losses_coral = []
aggre_losses_classifier = []
aggre_losses_classifier_tgt = []
aggre_losses_l2_src = []
aggre_losses_kld_src = []
aggre_losses_l2_tgt = []
aggre_losses_kld_tgt = []

aggre_losses_valid = []
aggre_losses_l2_valid = []
aggre_losses_classifier_valid = []
aggre_losses_classifier_valid_tgt = []
aggre_losses_coral_valid = []
aggre_losses_l2_src_valid = []
aggre_losses_kld_src_valid = []
aggre_losses_l2_tgt_valid = []
aggre_losses_kld_tgt_valid = []

# aggre_losses_l2_ddm = []

aggre_train_acc = []
aggre_test_acc = []
aggre_valid_acc = []
aggre_train_tgt_acc = []

n_epochs = 5
DDM_NUM = 20
DIFFERECE_COL = 5

# train the model
def train_model(train_dat, valid_dat, test_dat, model, device):
    model.train()
    # define the optimization
    criterion = CrossEntropyLoss()
    l2loss = MSELoss()
    l2loss_mse = F.mse_loss
    # optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
    # optimizer = torch.optim.SGD([{'params': model.feature.parameters()},
    #                              {'params':model.fc.parameters(),'lr':10*args.lr}],
    #                             lr= args.lr,momentum=args.momentum,weight_decay=args.weight_clay)

    # optimizer = RMSprop(model.parameters(), lr=0.0001)
    # optimizer1 = Adam([{'params': model.ddm.parameters(), 'lr': 0.01}], lr=0.01)
    # optimizer2 = RMSprop([{'params': model.feature.parameters()}, {'params': model.fc.parameters(), 'lr': 0.001}], lr=0.0001)
    # optimizer = Adam([{'params': model.ddm.parameters(), 'lr': 0.001}, {'params': model.feature.parameters()}, {'params': model.fc.parameters(), 'lr': 0.000005}], lr=0.000005)
    optimizer = Adam(model.parameters(), lr=0.001)

    es = EarlyStopping(patience=50)

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
        epoch_loss_l2_src = 0
        epoch_loss_kld_src = 0
        epoch_loss_l2_tgt = 0
        epoch_loss_kld_tgt = 0

        i = 0
        for it, (src_data, src_label, tgt_data, tgt_label) in enumerate(train_dat):
            # clear the gradients
            # optimizer1.zero_grad()
            # optimizer2.zero_grad()
            # src_data = src_data.reshape(int(src_data.shape[0] / 5), 5, src_data.shape[1])
            # src_data = src_data.permute(0, 2, 1)
            # tgt_data = tgt_data.reshape(int(tgt_data.shape[0] / 5), 5, tgt_data.shape[1])
            # tgt_data = tgt_data.permute(0, 2, 1)
            src_label = src_label[2:src_label.shape[0]:5]
            tgt_label = tgt_label[2:tgt_label.shape[0]:5]
            optimizer.zero_grad()
            if torch.cuda.is_available():
                tgt_data = tgt_data.to(device)
                src_data = src_data.to(device)
                src_label = src_label.to(device)
                tgt_label = tgt_label.to(device)

            # compute the model output
            # yhat = model(inputs)
            src_out, tgt_out, dm_out, centr1, centr2, decoded_src, mu_src, logvar_src, decoded_tgt, mu_tgt, logvar_tgt, ddmed_tgt, z_src, z_tgt = model(
                src_data, tgt_data)

            # vae loss on src
            src_data_cut = src_data[2:src_data.shape[0]:5]
            tgt_data_cut = tgt_data[2:tgt_data.shape[0]:5]
            loss_l2_src = l2loss_mse(decoded_src, src_data_cut, reduction='none')
            loss_l2_src = loss_l2_src.sum(axis=1)
            loss_l2_src = loss_l2_src.mean()  # average over batch dimension

            KLD_src = 0.5 * torch.sum(logvar_src.exp() - logvar_src - 1 + mu_src.pow(2), axis=1)
            batchsize_src = KLD_src.size(0)
            KLD_src = KLD_src.mean()

            # vae loss on tgt
            ddmed_tgt = ddmed_tgt[2:ddmed_tgt.shape[0]:5]
            loss_l2_tgt = l2loss_mse(decoded_tgt, ddmed_tgt, reduction='none')
            loss_l2_tgt = loss_l2_tgt.sum(axis=1)
            loss_l2_tgt = loss_l2_tgt.mean()  # average over batch dimension

            KLD_tgt = 0.5 * torch.sum(logvar_tgt.exp() - logvar_tgt - 1 + mu_tgt.pow(2), axis=1)
            batchsize_tgt = KLD_tgt.size(0)
            KLD_tgt = KLD_tgt.mean()

            # calculate loss
            # # loss = criterion(yhat, targets)
            # # epoch_loss = loss
            # print("src_out.shape:", src_out.shape)
            # print(src_out)
            # print("src_label.shape[0]:", len(src_label))
            # # print("src_label.shape[1]:", src_label.shape[1])
            # print(src_label.cpu().numpy())
            loss_classifier = criterion(src_out, src_label)
            # print("src_label:")
            # print(src_label)

            # loss_classifier_tgt = criterion(tgt_out, tgt_label)
            wgt = setWeight(src_label.cpu(), tgt_label.cpu())
            wgt = wgt.to(device)
            loss_classifier_tgt_prew = criterion(tgt_out, src_label)
            loss_classifier_tgt = (loss_classifier_tgt_prew * wgt / wgt.sum()).sum()

            # equalRate(src_label.cpu(), tgt_label.cpu())
            # print("tgt_label:")
            # print(tgt_label)
            # loss_coral = CORAL(src_out, tgt_out)
            # loss_coral = CORAL(centr1, centr2)

            # calculate MMD loss
            mmdLoss = MMD_loss(kernel_type='rbf')
            loss_mmd = mmdLoss(z_src, z_tgt)
            loss_coral = loss_mmd

            loss_l2 = l2loss(dm_out, src_data[:, 0:DDM_NUM + DIFFERECE_COL])
            sum_loss = loss_classifier + lambda_l2 * loss_l2 + loss_classifier_tgt.mean() + loss_l2_src + KLD_src + loss_l2_tgt + KLD_tgt + loss_mmd * 0.5

            # sum_loss = lambda_ * loss_coral + loss_classifier + lambda_l2 * loss_l2 + loss_classifier_tgt * 0.5 + loss_l2_src + KLD_src + loss_l2_tgt + KLD_tgt
            epoch_loss += sum_loss.item()
            epoch_loss_l2 += loss_l2.item()
            epoch_loss_classifier += loss_classifier.item()
            epoch_loss_classifier_tgt += loss_classifier_tgt.item()
            epoch_loss_coral += loss_coral.item()
            epoch_loss_l2_src += loss_l2_src.item()
            epoch_loss_kld_src += KLD_src.item()
            epoch_loss_l2_tgt += loss_l2_tgt.item()
            epoch_loss_kld_tgt += KLD_tgt.item()

            # credit assignment
            sum_loss.backward()
            # loss_l2.backward(retain_graph=True)
            # sum_loss.backward(retain_graph=True)
            # loss_l2.backward()
            # update model weights
            # optimizer1.step()
            # optimizer2.step()
            optimizer.step()
            i = i + 1

        print('DA Train Epoch: {:2d} [{:2d}/{:2d}]\t'
              'Lambda: {:.4f}, Class: {:.6f}, CORAL: {:.6f}, l2_loss: {:.6f}, vae_l2_src:{:.6f}, vae_kld_src:{:.6f}, vae_l2_tgt:{:.6f}, vae_kld_tgt:{:.6f} Total_Loss: {:.6f}'.format(
            epoch,
            i + 1,
            train_steps,
            lambda_,
            loss_classifier.item(),
            loss_coral.item(),
            loss_l2.item(),
            loss_l2_src.item(),
            KLD_src.item(),
            loss_l2_tgt.item(),
            KLD_tgt.item(),
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
        test_acc = evaluate_model_tgt(test_dat, model, device)
        aggre_test_acc.append(test_acc)
        print('DA test_acc: %.3f' % test_acc)

        epoch_loss = epoch_loss / train_steps
        aggre_losses.append(epoch_loss)
        print(f'DA epoch: {j:3} sum loss: {epoch_loss:6.4f}')

        epoch_loss_l2 = epoch_loss_l2 / train_steps
        aggre_losses_l2.append(epoch_loss_l2)
        print(f'DA epoch: {j:3} l2 loss: {epoch_loss_l2:6.4f}')

        epoch_loss_classifier = epoch_loss_classifier / train_steps
        aggre_losses_classifier.append(epoch_loss_classifier)
        print(f'DA epoch: {j:3} classifier src loss: {epoch_loss_classifier:6.4f}')

        epoch_loss_classifier_tgt = epoch_loss_classifier_tgt / train_steps
        aggre_losses_classifier_tgt.append(epoch_loss_classifier_tgt)
        print(f'DA epoch: {j:3} classifier tgt loss: {epoch_loss_classifier_tgt:6.4f}')

        epoch_loss_coral = epoch_loss_coral / train_steps
        aggre_losses_coral.append(epoch_loss_coral)
        print(f'DA epoch: {j:3} mmd loss: {epoch_loss_coral:6.4f}')

        epoch_loss_l2_src = epoch_loss_l2_src / train_steps
        aggre_losses_l2_src.append(epoch_loss_l2_src)
        print(f'DA epoch: {j:3} vae l2 src loss: {epoch_loss_l2_src:6.4f}')

        epoch_loss_kld_src = epoch_loss_kld_src / train_steps
        aggre_losses_kld_src.append(epoch_loss_kld_src)
        print(f'DA epoch: {j:3} vae kld src loss: {epoch_loss_kld_src:6.4f}')

        epoch_loss_l2_tgt = epoch_loss_l2_tgt / train_steps
        aggre_losses_l2_tgt.append(epoch_loss_l2_tgt)
        print(f'DA epoch: {j:3} vae l2 tgt loss: {epoch_loss_l2_tgt:6.4f}')

        epoch_loss_kld_tgt = epoch_loss_kld_tgt / train_steps
        aggre_losses_kld_src.append(epoch_loss_kld_tgt)
        print(f'DA epoch: {j:3} vae kld tgt loss: {epoch_loss_kld_tgt:6.4f}')

        # # calculate validate accuracy
        epoch_loss_valid, epoch_loss_l2_valid, epoch_loss_classifier_valid, epoch_loss_coral_valid, epoch_loss_classifier_valid_tgt, epoch_loss_l2_src_valid, epoch_loss_kld_src_valid, epoch_loss_l2_tgt_valid, epoch_loss_kld_tgt_valid = evaluate_model_stop(
            valid_dat, model, device)  # evalution on dev set (i.e., holdout from training)
        aggre_losses_valid.append(epoch_loss_valid)
        aggre_losses_l2_valid.append(epoch_loss_l2_valid)
        aggre_losses_classifier_valid.append(epoch_loss_classifier_valid)
        aggre_losses_classifier_valid_tgt.append(epoch_loss_classifier_valid_tgt)
        aggre_losses_coral_valid.append(epoch_loss_coral_valid)
        aggre_losses_l2_src_valid.append(epoch_loss_l2_src_valid)
        aggre_losses_kld_src_valid.append(epoch_loss_kld_src_valid)
        aggre_losses_l2_tgt_valid.append(epoch_loss_l2_tgt_valid)
        aggre_losses_kld_tgt_valid.append(epoch_loss_kld_tgt_valid)

        print(f'DA epoch: {j:3} valid sum loss: {epoch_loss_valid:6.4f}')
        print(f'DA epoch: {j:3} valid l2 loss: {epoch_loss_l2_valid:6.4f}')
        print(f'DA epoch: {j:3} valid classifier loss: {epoch_loss_classifier_valid:6.4f}')
        print(f'DA epoch: {j:3} valid tgt classifier loss: {epoch_loss_classifier_valid_tgt:6.4f}')
        print(f'DA epoch: {j:3} valid mmd loss: {epoch_loss_coral_valid:6.4f}')
        print(f'DA epoch: {j:3} valid vae l2 src loss: {epoch_loss_l2_src_valid:6.4f}')
        print(f'DA epoch: {j:3} valid vae kld src loss: {epoch_loss_kld_src_valid:6.4f}')
        print(f'DA epoch: {j:3} valid vae l2 tgt loss: {epoch_loss_l2_tgt_valid:6.4f}')
        print(f'DA epoch: {j:3} valid vae kld tgt loss: {epoch_loss_kld_tgt_valid:6.4f}')

        # if es.step(np.array(epoch_loss_classifier_valid_tgt)):
        if es.step(np.array(epoch_loss_valid)):
            print(f'Early Stopping Criteria Met!')
            break  # early stop criterion is met, we can stop now


# evaluate the validation loss for early stop
def evaluate_model_stop(valid_dl, model, device):
    model.eval()
    # define the optimization
    criterion = CrossEntropyLoss()
    l2loss = MSELoss()
    l2loss_mse = F.mse_loss
    epoch_loss = 0
    epoch_loss_l2 = 0
    epoch_loss_classifier = 0
    epoch_loss_classifier_valid = 0
    epoch_loss_coral = 0
    epoch_loss_l2_src = 0
    epoch_loss_kld_src = 0
    epoch_loss_l2_tgt = 0
    epoch_loss_kld_tgt = 0

    valid_steps = len(valid_dl)
    print("DA valid_steps:", valid_steps)

    for i, (src_data, src_label, tgt_data, tgt_label) in enumerate(valid_dl):
        # src_data = src_data.reshape(int(src_data.shape[0] / 5), 5, src_data.shape[1])
        # src_data = src_data.permute(0, 2, 1)
        # tgt_data = tgt_data.reshape(int(tgt_data.shape[0] / 5), 5, tgt_data.shape[1])
        # tgt_data = tgt_data.permute(0, 2, 1)
        src_label = src_label[2:src_label.shape[0]:5]
        tgt_label = tgt_label[2:tgt_label.shape[0]:5]
        if torch.cuda.is_available():
            tgt_data = tgt_data.to(device)
            src_data = src_data.to(device)
            src_label = src_label.to(device)
            tgt_label = tgt_label.to(device)

        with torch.no_grad():
            src_out, tgt_out, dm_out, centr1, centr2, decoded_src, mu_src, logvar_src, decoded_tgt, mu_tgt, logvar_tgt, ddmed_tgt, z_src, z_tgt = model(
                src_data, tgt_data)
        # calculate loss
        # vae loss on src
        src_data_cut = src_data[2:src_data.shape[0]:5]
        loss_l2_src = l2loss_mse(decoded_src, src_data_cut, reduction='none')
        loss_l2_src = loss_l2_src.sum(axis=1)
        loss_l2_src = loss_l2_src.mean()  # average over batch dimension

        KLD_src = 0.5 * torch.sum(logvar_src.exp() - logvar_src - 1 + mu_src.pow(2), axis=1)
        batchsize_src = KLD_src.size(0)
        KLD_src = KLD_src.mean()

        # vae loss on tgt
        ddmed_tgt = ddmed_tgt[2:ddmed_tgt.shape[0]:5]
        loss_l2_tgt = l2loss_mse(decoded_tgt, ddmed_tgt, reduction='none')
        loss_l2_tgt = loss_l2_tgt.sum(axis=1)
        loss_l2_tgt = loss_l2_tgt.mean()  # average over batch dimension

        KLD_tgt = 0.5 * torch.sum(logvar_tgt.exp() - logvar_tgt - 1 + mu_tgt.pow(2), axis=1)
        batchsize_tgt = KLD_tgt.size(0)
        KLD_tgt = KLD_tgt.mean()

        loss_classifier = criterion(src_out, src_label)
        # loss_classifier_valid = criterion(tgt_out, tgt_label)
        loss_classifier_valid = criterion(tgt_out, src_label)
        # loss_coral = CORAL(src_out, tgt_out)
        # loss_coral = CORAL(centr1, centr2)
        # loss_coral = KLD_tgt

        # calculate MMD loss
        mmdLoss = MMD_loss(kernel_type='rbf')
        loss_mmd = mmdLoss(z_src, z_tgt)
        loss_coral = loss_mmd

        loss_l2 = l2loss(dm_out, src_data[:, 0:DDM_NUM + DIFFERECE_COL])
        sum_loss = loss_classifier + lambda_l2 * loss_l2 + loss_classifier_valid + loss_l2_src + KLD_src + loss_l2_tgt + KLD_tgt + loss_mmd * 0.5
        epoch_loss += sum_loss.item()
        epoch_loss_l2 += loss_l2.item()
        epoch_loss_classifier += loss_classifier.item()
        epoch_loss_classifier_valid += loss_classifier_valid.item()
        epoch_loss_coral += loss_coral.item()
        epoch_loss_l2_src += loss_l2_src.item()
        epoch_loss_kld_src += KLD_src.item()
        epoch_loss_l2_tgt += loss_l2_tgt.item()
        epoch_loss_kld_tgt += KLD_tgt.item()

    return epoch_loss / valid_steps, epoch_loss_l2 / valid_steps, epoch_loss_classifier / valid_steps, epoch_loss_coral / valid_steps, epoch_loss_classifier_valid / valid_steps, epoch_loss_l2_src / valid_steps, epoch_loss_kld_src / valid_steps, epoch_loss_l2_tgt / valid_steps, epoch_loss_kld_tgt / valid_steps


# evaluate the model source
def evaluate_model_src(test_dl, model, device):
    model.eval()
    predictions, actuals = list(), list()
    # test_steps = len(test_dl)
    # iter_test = iter(test_dl)
    for i, (src_data, src_label, tgt_data, tgt_label) in enumerate(test_dl):
        # for i in range(test_steps):
        # evaluate the model on the test set
        # tgt_data, targets = iter_test.next()
        # src_data = src_data.reshape(int(src_data.shape[0] / 5), 5, src_data.shape[1])
        # src_data = src_data.permute(0, 2, 1)
        # tgt_data = tgt_data.reshape(int(tgt_data.shape[0] / 5), 5, tgt_data.shape[1])
        # tgt_data = tgt_data.permute(0, 2, 1)
        src_label = src_label[2:src_label.shape[0]:5]
        tgt_label = tgt_label[2:tgt_label.shape[0]:5]
        if torch.cuda.is_available():
            src_data = src_data.to(device)
            src_label = src_label.to(device)

        tgt_data = src_data
        targets = src_label
        # temp = torch.zeros((tgt_data.shape[0], DIFFERECE_COL))
        # tmp_data = torch.cat((tgt_data, temp), 1)
        with torch.no_grad():
            yhat, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(tgt_data, tgt_data[:, 0:26])
        # retrieve numpy arrayx
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
        # src_data = src_data.reshape(int(src_data.shape[0] / 5), 5, src_data.shape[1])
        # src_data = src_data.permute(0, 2, 1)
        # tgt_data = tgt_data.reshape(int(tgt_data.shape[0] / 5), 5, tgt_data.shape[1])
        # tgt_data = tgt_data.permute(0, 2, 1)
        src_label = src_label[2:src_label.shape[0]:5]
        # tgt_label = tgt_label[2:tgt_label.shape[0]:5]
        if torch.cuda.is_available():
            target_data = target_data.to(device)
            target_label = target_label.to(device)

        tgt_data = target_data
        # targets = target_label # for 3 label ues this TODO: verify DAMA to use the src_label instead of target_label
        targets = src_label

        temp = torch.zeros((tgt_data.shape[0], DIFFERECE_COL))
        temp = temp.to(device)
        tmp_data = torch.cat((tgt_data, temp), 1)
        # temp = torch.zeros((tgt_data.shape[0], DIFFERECE_COL))
        # tmp_data = torch.cat((tgt_data, temp), 1)
        with torch.no_grad():
            _, yhat, _, _, _, _, _, _, _, _, _, _, _, _ = model(tmp_data, tgt_data)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path")
    parser.add_argument("--model_saving_path")
    args = parser.parse_args()

    NUM_LALBELS = 3
    EPOCHS = 50
    lambda_ = 0.001
    lambda_l2 = 0.05
    BATCH_SIZE = 250

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("torch.cuda.is_available:", torch.cuda.is_available())

    # load the training data
    train_dat, valid_dat = preProcessing(args.training_data_path, args.model_saving_path, b_size=BATCH_SIZE)

    # initiate the model
    # model = Deep_coral(num_classes=NUM_LALBELS)
    model = Deep_VAE(num_classes=NUM_LALBELS)
    # train the model
    # train_model(train_dat, valid_dat, model, EPOCHS, lambda_, lambda_l2, _device)
    train_model(train_dat, valid_dat, valid_dat, model, _device)

    # evaluate the model on valid data
    acc = evaluate_model_tgt(valid_dat, model, _device)
    print('Accuracy: %.3f' % acc)

    # save the model
    torch.save(model, args.model_saving_path + 'model.pth')

