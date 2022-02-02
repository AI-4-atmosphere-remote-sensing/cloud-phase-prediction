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
  model.load_state_dict(torch.load(args.model_saving_path + '/model_region.pt'))
  model.to(_device)
  sc_X = load(open(args.model_saving_path + '/scaler_region.pkl', 'rb'))

  # evaluate on the testing data
  test_files = glob.glob(args.testing_data_path + '/*.npz')
  for test_file in test_files:
    X_s_test, Y_s_test, X_t_test, Y_t_test = load_test_data(test_file, sc_X)
    test_dat = prepare_data(X_s_test, Y_s_test, X_t_test, Y_t_test, BATCH_SIZE)
    acc = evaluate_model_tgt(test_dat, model, _device)
    print("Test on file:", test_file)
    print('Accuracy is: %.3f' % acc)








