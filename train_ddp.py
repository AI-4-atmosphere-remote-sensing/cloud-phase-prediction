'''
    Notices: “Copyright © 2022 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.   All Rights Reserved.”

    Disclaimer: 
    No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
    Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.

'''

import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from numpy import vstack
from numpy import argmax
from torch.optim import SGD,RMSprop,Adam
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from data_utils import preProcessing
from model import Deep_coral

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import time

#import time
#import socket
#from torch.utils.data import Dataset, DataLoader


def ddp_setup(rank, world_size):
    #print("local_host need to be connected")
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "6006" 
    
    #print("connected to localhost")
    #start_time = time.time()
    # Connect to localhost on port 80
    #sock = socket.create_connection(("127.0.0.1", 6006)
    #end_time = time.time()
    # Calculate and print the time taken to connect
    #time_taken = end_time - start_time
    #print("Time taken to connect: {:.5f} seconds".format(time_taken))
   
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


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
    
    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dat,num_replicas=2,
    #rank=0)
    #train_loader = torch.utils.data.DataLoader(train_dat,batch_size=2048,sampler=train_sampler)
    #model.to(device)
    if torch.cuda.is_available():
      model = model.cuda()
    model = DDP(model, device_ids=[device])
    #model.train()
    # define the optimization
    criterion = torch.nn.CrossEntropyLoss()
    l2loss = MSELoss()
    # optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
    # optimizer = torch.optim.SGD([{'params': model.feature.parameters()},
    #                              {'params':model.fc.parameters(),'lr':10*args.lr}],
    #                             lr= args.lr,momentum=args.momentum,weight_decay=args.weight_clay)

    # optimizer = RMSprop(model.parameters(), lr=0.0001)
    # optimizer1 = Adam([{'params': model.ddm.parameters(), 'lr': 0.01}], lr=0.01)
    # optimizer2 = RMSprop([{'params': model.feature.parameters()}, {'params': model.fc.parameters(), 'lr': 0.001}], lr=0.0001)
    #optimizer = Adam([{'params': model.ddm.parameters(), 'lr': 0.001}, {'params': model.feature.parameters()}, {'params': model.fc.parameters(), 'lr': 0.000005}], lr=0.000005)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    es = EarlyStopping(patience=10)

    
    #model = DDP(model, device_ids=[device])
    b_sz = len(next(iter(train_dat))[0])
    # enumerate epochs for DA
    j = 0
    times = []
    torch.cuda.synchronize()
    start_epoch = time.time()
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
        print(f"[GPU{torch.cuda.current_device()}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_dat)}")

        for i, (src_data, src_label, tgt_data, tgt_label) in enumerate(train_dat):
            # clear the gradients
            # optimizer1.zero_grad()
            # optimizer2.zero_grad()
            #optimizer.zero_grad()
            if torch.cuda.is_available():
              tgt_data = tgt_data.to(device)
              src_data = src_data.to(device)
              src_label = src_label.to(device)
              tgt_label = tgt_label.to(device)

            # compute the model output
            src_out, tgt_out, dm_out, centr1, centr2 = model(src_data, tgt_data)

            # calculate loss
            loss_classifier = criterion(src_out, src_label)
            loss_classifier_tgt = criterion(tgt_out, tgt_label)
            loss_coral = CORAL(centr1, centr2)

            loss_l2 = l2loss(dm_out, src_data[:, 0:25])
            sum_loss = lambda_ * loss_coral + loss_classifier + lambda_l2 * loss_l2 + loss_classifier_tgt * 0.5
            epoch_loss += sum_loss.item()
            epoch_loss_l2 += loss_l2.item()
            epoch_loss_classifier += loss_classifier.item()
            epoch_loss_classifier_tgt += loss_classifier_tgt.item()
            epoch_loss_coral += loss_coral.item()
            
            optimizer.zero_grad()
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
    model.train()
    torch.cuda.synchronize()
    end_epoch = time.time()
    elapsed = end_epoch - start_epoch
    times.append(elapsed)
    avg_time = sum(times)/n_epochs
    print(avg_time)

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

    for i, (src_data, src_label, tgt_data, tgt_label) in enumerate(test_dl):
        if torch.cuda.is_available():
          src_data = src_data.to(device)
          src_label = src_label.to(device)

        tgt_data = src_data
        targets = src_label

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
def evaluate_model_tgt(test_dl, model, device):
    model.eval()
    predictions, actuals = list(), list()

    for i, (src_data, src_label, target_data, target_label) in enumerate(test_dl):
        if torch.cuda.is_available():
          target_data = target_data.to(device)
          target_label = target_label.to(device)

        tgt_data = target_data
        targets = target_label
        DIFFERECE_COL = 5
        temp = torch.zeros((tgt_data.shape[0], DIFFERECE_COL))
        temp = temp.to(device)
        tmp_data = torch.cat((tgt_data, temp), 1)
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


def main(rank, world_size,training_data_path,model_saving_path, BATCH_SIZE):
    
    ddp_setup(rank, world_size)
    print("DDP_SETUP completed successfully")

    #dataset, model, optimizer = load_train_objs()
    #train_data = prepare_dataloader(dataset, batch_size)
    #trainer = Trainer(model, train_data, optimizer, rank, save_every)
    #trainer.train(total_epochs)
    # load the training data
    train_dat, valid_dat = preProcessing(training_data_path, model_saving_path, b_size=BATCH_SIZE)
    NUM_LALBELS = 3
    EPOCHS = 50
    lambda_ = 0.001
    lambda_l2 = 0.05
    BATCH_SIZE = 2048

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("torch.cuda.is_available:", torch.cuda.is_available())
    # initiate the model
    model = Deep_coral(num_classes=NUM_LALBELS)
    # train the model
    train_model(train_dat, valid_dat, model, EPOCHS, lambda_, lambda_l2, _device)
    #print(train_model)
    # evaluate the model on valid data
    acc = evaluate_model_tgt(valid_dat, model, _device)
    print('Accuracy: %.3f' % acc)

    # save the model
    #torch.save(model.module.state_dict(), args.model_saving_path + 'model.pth')
    if hasattr(model, 'module'):
      torch.save(model.module.state_dict(), model_saving_path + 'model.pth')
    else:
      torch.save(model.state_dict(), model_saving_path + 'model.pth')

    destroy_process_group()



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--training_data_path")
  parser.add_argument("--model_saving_path")
  #parser.add_argument('EPOCHS', type=int, help='Total epochs to train the model')
  #parser.add_argument('save_every', type=int, help='How often to save a snapshot')
  #parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
  
  args = parser.parse_args()

  NUM_LALBELS = 3
  EPOCHS = 50
  lambda_ = 0.001
  lambda_l2 = 0.05
  BATCH_SIZE = 2048
  
  world_size = torch.cuda.device_count()
  print(world_size)
   
  mp.spawn(main, args=(world_size, args.training_data_path,args.model_saving_path, BATCH_SIZE), nprocs=world_size)
