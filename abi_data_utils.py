import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filename):
    """
    Load and preprocess the new .npz data structure directly.
    Args:
        filename (str): Path to the .npz file.
    Returns:
        numpy.ndarray: Processed features (X).
        numpy.ndarray: Processed labels (Y).
    """
    # Load the .npz file
    data = np.load(filename)

    # Extract CALIPSO data (combine all CALIPSO keys)
    calipso_keys = [key for key in data.keys() if key.startswith("CALIOP")]
    calipso_data = np.hstack([data[key] for key in calipso_keys])

    # Extract ABI data
    abi_data = data["ABI"]

    # Extract auxiliary data: latitude and longitude
    lat_lon_data = np.stack([data["ABI_Lat_1km"], data["ABI_Lon_1km"]], axis=1)

    # Combine CALIPSO, ABI, and lat/lon into a single feature matrix
    X = np.hstack([calipso_data, abi_data, lat_lon_data])

    # Extract labels (index 1 of "label" key)
    Y = data["label"][:, 1]

    # Standardize the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, Y

