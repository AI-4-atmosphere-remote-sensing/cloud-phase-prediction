import numpy as np
from netCDF4 import Dataset
import os

# Function to calculate the glint angle
def calculate_glint_angle(solar_zenith, solar_azimuth, view_zenith, view_azimuth):
    phi = np.deg2rad(solar_azimuth - view_azimuth)
    cos_glint = (np.cos(np.deg2rad(view_zenith)) * np.cos(np.deg2rad(solar_zenith)) -
                 np.sin(np.deg2rad(view_zenith)) * np.sin(np.deg2rad(solar_zenith)) * np.cos(phi))
    glint_angle = np.degrees(np.arccos(np.clip(cos_glint, -1.0, 1.0)))
    return glint_angle

# Function to apply the glint filter
def apply_glint_filter(glint_angle, threshold=40):
    return glint_angle > threshold

# Function to find matching prediction file (for predicted dataset)
def find_matching_prediction_file_pred(v03_filename, prediction_folder):
    base_name = os.path.basename(v03_filename).replace('.nc', '_prediction_onlyM16.nc')
    return os.path.join(prediction_folder, base_name)

# Function to find matching ground truth file (for GT dataset)
def find_matching_prediction_file_GT(v03_filename, prediction_folder):
    parts = v03_filename.split('.')
    date_part = parts[0].split('_')[1]  # Extract the date part
    time_part = parts[1]  # Extract the time part
    prediction_filename = f'CLDPROP_L2_VIIRS_SNPP.{date_part}.{time_part}.nc'
    return os.path.join(prediction_folder, prediction_filename)

# Base paths
v03_base_folder = '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/VNP03MOD/2017/'
prediction_base_folder = '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/predict_2016/2017/'
GT_base_folder = '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/ctphase/CLDPROP_L2_VIIRS_SNPP/2017/'

# Days to process
days = ['{:03d}'.format(d) for d in range(1, 32)]

# Initialize aggregate grids
grid_size = (180, 360)

# Prediction grids
total_pixels_pred = np.zeros(grid_size, dtype=int)
liquid_cloud_pixels_pred = np.zeros(grid_size, dtype=int)
ice_cloud_pixels_pred = np.zeros(grid_size, dtype=int)

# Ground truth grids
total_pixels_GT = np.zeros(grid_size, dtype=int)
liquid_cloud_pixels_GT = np.zeros(grid_size, dtype=int)
ice_cloud_pixels_GT = np.zeros(grid_size, dtype=int)

# Define the x-axis range
x_min, x_max = 1100, 2100

# Iterate over each day
for day in days:
    print(f"Processing day {day}")

    v03_folder = os.path.join(v03_base_folder, day)
    pred_folder = os.path.join(prediction_base_folder, day)
    GT_folder = os.path.join(GT_base_folder, day)

    # Iterate over files in VNP03 folder
    i = 1
    for v03_file in os.listdir(v03_folder):
        if not v03_file.endswith('.nc'):
            continue
        
        print(f" Day {day}, File {i}: {v03_file}")
        i += 1

        file_v03 = os.path.join(v03_folder, v03_file)
        file_pred = find_matching_prediction_file_pred(v03_file, pred_folder)
        file_GT = find_matching_prediction_file_GT(v03_file, GT_folder)

        # Check that both prediction and GT files exist
        if not os.path.exists(file_pred):
            print(f" Prediction file not found: {file_pred}")
            continue
        if not os.path.exists(file_GT):
            print(f" GT file not found: {file_GT}")
            continue

        # === Read geolocation and angles ===
        with Dataset(file_v03, 'r') as nc:
            lon = nc['/geolocation_data/longitude'][:, x_min:x_max]
            lat = nc['/geolocation_data/latitude'][:, x_min:x_max]
            solar_zenith = nc['/geolocation_data/solar_zenith'][:, x_min:x_max]
            solar_azimuth = nc['/geolocation_data/solar_azimuth'][:, x_min:x_max]
            view_zenith = nc['/geolocation_data/sensor_zenith'][:, x_min:x_max]
            view_azimuth = nc['/geolocation_data/sensor_azimuth'][:, x_min:x_max]

        # === Read prediction data ===
        with Dataset(file_pred, 'r') as nc:
            prediction = nc.variables['prediction'][:, x_min:x_max]

        # === Read ground truth data ===
        with Dataset(file_GT, 'r') as nc:
            if 'Cloud_Phase_Cloud_Top_Properties' in nc.groups['geophysical_data'].variables:
                GT_prediction = nc.groups['geophysical_data'].variables['Cloud_Phase_Cloud_Top_Properties'][:, x_min:x_max]
            else:
                print(f" 'Cloud_Phase_Cloud_Top_Properties' not found in {file_GT}")
                continue

        # === Apply filters ===
        glint_angle = calculate_glint_angle(solar_zenith, solar_azimuth, view_zenith, view_azimuth)
        glint_filter = apply_glint_filter(glint_angle)

        valid_pixels = (solar_zenith <= 83) & glint_filter
        if not np.any(valid_pixels):
            print(f" No valid pixels after filtering for {v03_file}")
            continue

        # === Subset lat, lon, prediction, and GT data to valid pixels ===
        lat_valid = lat[valid_pixels]
        lon_valid = lon[valid_pixels]
        pred_valid = prediction[valid_pixels]
        GT_valid = GT_prediction[valid_pixels]

        # === Compute common grid indices ===
        grid_lat = np.clip(np.floor(lat_valid).astype(int) + 90, 0, 179)
        grid_lon = np.clip(np.floor(lon_valid).astype(int) + 180, 0, 359)

        # === Update Prediction counters ===
        np.add.at(total_pixels_pred, (grid_lat, grid_lon), 1)
        np.add.at(liquid_cloud_pixels_pred, (grid_lat[pred_valid == 1], grid_lon[pred_valid == 1]), 1)
        np.add.at(ice_cloud_pixels_pred, (grid_lat[pred_valid == 2], grid_lon[pred_valid == 2]), 1)

        # === Update Ground Truth counters ===
        np.add.at(total_pixels_GT, (grid_lat, grid_lon), 1)
        np.add.at(liquid_cloud_pixels_GT, (grid_lat[GT_valid == 1], grid_lon[GT_valid == 1]), 1)
        np.add.at(ice_cloud_pixels_GT, (grid_lat[GT_valid == 2], grid_lon[GT_valid == 2]), 1)

# === Calculate cloud fractions ===
cloud_fraction_pred = np.full(grid_size, np.nan, dtype=float)
cloud_fraction_GT = np.full(grid_size, np.nan, dtype=float)

# Only calculate where there are valid counts
valid_counts_pred = total_pixels_pred > 0
valid_counts_GT = total_pixels_GT > 0

cloud_fraction_pred[valid_counts_pred] = (liquid_cloud_pixels_pred[valid_counts_pred] + ice_cloud_pixels_pred[valid_counts_pred]) / total_pixels_pred[valid_counts_pred]
cloud_fraction_GT[valid_counts_GT] = (liquid_cloud_pixels_GT[valid_counts_GT] + ice_cloud_pixels_GT[valid_counts_GT]) / total_pixels_GT[valid_counts_GT]

# === Save cloud fraction results ===
np.save('/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/cloud_fraction2017_31daysNEW.npy', cloud_fraction_pred)
np.save('/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/cloud_fraction_GT2017_31daysNEW.npy', cloud_fraction_GT)

print("Cloud fraction processing complete. Files saved.")

