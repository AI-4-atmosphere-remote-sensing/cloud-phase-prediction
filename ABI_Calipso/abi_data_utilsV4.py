#ABI_data_utilsV4.py using new data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pickle import dump
import glob

def loadData(filename):
    data = np.load(filename)

    # Load ABI-CALIPSO dataset
    latlon = data['latlon']  # (N, 2)
    angles = data['angles']  # (N, 4)
    X_a = data['abi']        # (N, 16)
    Y_c = data['label'][:, 1]  # Use only CALIPSO label index 1 and ABI index 0
    X_c = data['calipso']    # (N, 20)

    # --- Extract angles ---
    solar_zenith = angles[:, 0]
    solar_azimuth = angles[:, 1]
    view_zenith = angles[:, 2]
    view_azimuth = angles[:, 3]

    # --- Solar Zenith Filter: between 0 and 83 degrees ---
    solar_filter = (solar_zenith >= 0) & (solar_zenith <= 83)

    # --- Glint Angle Filter: glint_angle > 40 degrees ---
    phi = np.deg2rad(solar_azimuth - view_azimuth)
    cos_glint = (np.cos(np.deg2rad(view_zenith)) * np.cos(np.deg2rad(solar_zenith)) -
                 np.sin(np.deg2rad(view_zenith)) * np.sin(np.deg2rad(solar_zenith)) * np.cos(phi))
    glint_angle = np.degrees(np.arccos(np.clip(cos_glint, -1.0, 1.0)))
    glint_filter = glint_angle > 40

    # --- Combined Filter: solar + glint ---
    valid_rows = solar_filter & glint_filter

    # Apply filtering
    X_a, X_c, Y_c = X_a[valid_rows], X_c[valid_rows], Y_c[valid_rows]
    latlon, angles = latlon[valid_rows], angles[valid_rows]

    # Filter out CALIPSO label == 1
    valid_rows = Y_c != 1
    X_a, X_c, Y_c = X_a[valid_rows], X_c[valid_rows], Y_c[valid_rows]
    latlon, angles = latlon[valid_rows], angles[valid_rows]

    # Remap labels: 0 → 0, 2 → 1, 3 → 2
    label_mapping = {0: 0, 2: 1, 3: 2}
    Y_c = np.vectorize(label_mapping.get)(Y_c)
    Y_a = Y_c.copy()  # Use filtered CALIPSO label for ABI as well

    # Append latlon and angles to CALIPSO features
    X_c = np.concatenate((X_c, latlon, angles), axis=1)

    # Replace NaNs
    X_a = np.nan_to_num(X_a)
    X_c = np.nan_to_num(X_c)

    return X_a, X_c, Y_a, Y_c

def preProcessing(training_data_path, model_saving_path, b_size=2048):
    train_files = glob.glob(training_data_path + '/*.npz')

    # Load and concatenate data from multiple files
    X_a, X_c, Y_a, Y_c = loadData(train_files[0])
    for train_file in train_files[1:]:
        X_a_new, X_c_new, Y_a_new, Y_c_new = loadData(train_file)
        X_a = np.concatenate((X_a, X_a_new), axis=0)
        X_c = np.concatenate((X_c, X_c_new), axis=0)
        Y_a = np.concatenate((Y_a, Y_a_new), axis=0)
        Y_c = np.concatenate((Y_c, Y_c_new), axis=0)

    # Merge ABI and CALIPSO labels (using the filtered Y_c for both)
    Y = np.column_stack((Y_a, Y_c))

    # Combine ABI and CALIPSO for standardization
    X = np.concatenate((X_a, X_c), axis=1)

    # **Feature Standardization**
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    # Save the scaler
    dump(sc_X, open(model_saving_path + '/scaler.pkl', 'wb'))

    # Split into training and validation sets
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

    # Separate standardized ABI and CALIPSO data
    x_train_a, x_train_c = x_train[:, :16], x_train[:, 16:]
    x_valid_a, x_valid_c = x_valid[:, :16], x_valid[:, 16:]

    # Prepare PyTorch DataLoaders
    train_dl = DataLoader(TensorDataset(torch.tensor(x_train_a, dtype=torch.float32),
                                        torch.tensor(y_train[:, 1], dtype=torch.long),
                                        torch.tensor(x_train_c, dtype=torch.float32),
                                        torch.tensor(y_train[:, 0], dtype=torch.long)),
                                        batch_size=b_size, shuffle=True)

    valid_dl = DataLoader(TensorDataset(torch.tensor(x_valid_a, dtype=torch.float32),
                                        torch.tensor(y_valid[:, 1], dtype=torch.long),
                                        torch.tensor(x_valid_c, dtype=torch.float32),
                                        torch.tensor(y_valid[:, 0], dtype=torch.long)),
                                        batch_size=b_size, shuffle=False)

    return train_dl, valid_dl

