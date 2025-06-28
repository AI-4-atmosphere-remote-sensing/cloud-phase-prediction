"""
    Notices: “Copyright © 2022 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.   All Rights Reserved.”

    Disclaimer: 
    No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
    Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.

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
from data_utils_vdam import load_test_data
from data_utils_vdam import prepare_data
from model import Deep_VAE
from train_vdam import evaluate_model_tgt

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--testing_data_path")
  parser.add_argument("--model_saving_path")
  args = parser.parse_args()

  NUM_LALBELS = 3
  BATCH_SIZE = 250

  _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("torch.cuda.is_available:", torch.cuda.is_available())

  # load the trained model
  model = Deep_VAE(num_classes=NUM_LALBELS)
  model = torch.load(args.model_saving_path + '/model.pth')
  sc_X = load(open(args.model_saving_path + '/scaler.pkl', 'rb'))

  # evaluate on the testing data
  test_files = glob.glob(args.testing_data_path + '/*.npz')
  for test_file in test_files:
    X_s_test, Y_s_test, X_t_test, Y_t_test = load_test_data(test_file, sc_X)
    print(X_s_test.shape)
    test_len = int(X_s_test.shape[0] / BATCH_SIZE) * BATCH_SIZE
    print(test_len)
    X_s_test = X_s_test[0:test_len]
    Y_s_test = Y_s_test[0:test_len]
    X_t_test = X_t_test[0:test_len]
    Y_t_test = Y_t_test[0:test_len]
    test_dat = prepare_data(X_s_test, Y_s_test, X_t_test, Y_t_test, BATCH_SIZE)
    acc = evaluate_model_tgt(test_dat, model, _device)
    print("Test on file:", test_file)
    print('Accuracy is: %.3f' % acc)
