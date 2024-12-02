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

# Function to find matching prediction file
def find_matching_prediction_file(v03_filename, prediction_folder):
    parts = v03_filename.split('.')
    date_part = parts[0].split('_')[1]  # Extract the date part
    time_part = parts[1]  # Extract the time part
    prediction_filename = f'CLDPROP_L2_VIIRS_SNPP.{date_part}.{time_part}.nc'
    return os.path.join(prediction_folder, prediction_filename)

# Base path to the folders
v03_base_folder = '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/VNP03MOD/2017/'
prediction_base_folder = '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/ctphase/CLDPROP_L2_VIIRS_SNPP/2017/'

# List of folder names for the first seven days
days = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015','016','017','018','019','020','021','022','023','024','025','026','027','028','029','030','031']

# Initialize grids for counting total pixels and cloud pixels across all days
grid_size = (180, 360)
total_pixels_Gr = np.zeros(grid_size, dtype=int)
liquid_cloud_pixels_Gr = np.zeros(grid_size, dtype=int)
ice_cloud_pixels_Gr = np.zeros(grid_size, dtype=int)

# Define the x-axis range for cloud fraction calculation
x_min, x_max = 1100, 2100

# Iterate over the first seven days
for day in days:
    v03_folder = os.path.join(v03_base_folder, day)
    prediction_folder = os.path.join(prediction_base_folder, day)

    i = 1
    # Iterate over each file in the current day's VNP03 folder
    for v03_file in os.listdir(v03_folder):
        print(f"Processing day {day}, file {i}: {v03_file}")
        i += 1
        if v03_file.endswith('.nc'):
            file_v03 = os.path.join(v03_folder, v03_file)
            file_prediction = find_matching_prediction_file(v03_file, prediction_folder)

            # Check if the prediction file exists before proceeding
            if not os.path.exists(file_prediction):
                print(f"Prediction file not found for {v03_file}")
                continue

            with Dataset(file_v03, 'r') as nc:
                lon = nc['/geolocation_data/longitude'][:, x_min:x_max]
                lat = nc['/geolocation_data/latitude'][:, x_min:x_max]
                solar_zenith = nc['/geolocation_data/solar_zenith'][:, x_min:x_max]
                solar_azimuth = nc['/geolocation_data/solar_azimuth'][:, x_min:x_max]
                view_zenith = nc['/geolocation_data/sensor_zenith'][:, x_min:x_max]
                view_azimuth = nc['/geolocation_data/sensor_azimuth'][:, x_min:x_max]

            with Dataset(file_prediction, 'r') as nc:
                if 'Cloud_Phase_Cloud_Top_Properties' in nc.groups['geophysical_data'].variables:
                    cloud_phase_data = nc.groups['geophysical_data'].variables['Cloud_Phase_Cloud_Top_Properties'][:, x_min:x_max]
                    prediction = cloud_phase_data
                else:
                    print(f"'Cloud_Phase_Cloud_Top_Properties' variable not found in {file_prediction}")
                    continue

            # Calculate the glint angle and apply the glint filter
            glint_angle = calculate_glint_angle(solar_zenith, solar_azimuth, view_zenith, view_azimuth)
            glint_filter = apply_glint_filter(glint_angle)

            # Apply solar zenith filter and glint filter
            valid_pixels = (solar_zenith <= 83) & glint_filter
            lat = lat[valid_pixels]
            lon = lon[valid_pixels]
            prediction = prediction[valid_pixels]

            # Convert latitude and longitude to grid indices
            grid_lat = np.clip(np.floor(lat).astype(int) + 90, 0, 179)
            grid_lon = np.clip(np.floor(lon).astype(int) + 180, 0, 359)

            # Update counts using vectorized operations for the current file
            np.add.at(total_pixels_Gr, (grid_lat.flatten(), grid_lon.flatten()), 1)
            np.add.at(liquid_cloud_pixels_Gr, (grid_lat[prediction == 1].flatten(), grid_lon[prediction == 1].flatten()), 1)
            np.add.at(ice_cloud_pixels_Gr, (grid_lat[prediction == 2].flatten(), grid_lon[prediction == 2].flatten()), 1)

# Calculate the aggregate cloud fraction for liquid and ice clouds
valid_counts_Gr = total_pixels_Gr > 0
cloud_fraction_Gr = np.full(grid_size, np.nan, dtype=float)
cloud_fraction_Gr[valid_counts_Gr] = (liquid_cloud_pixels_Gr[valid_counts_Gr] + ice_cloud_pixels_Gr[valid_counts_Gr]) / total_pixels_Gr[valid_counts_Gr]

# Save the aggregated cloud_fraction_Gr to a file
np.save('/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/cloud_fraction_GT2017_31days.npy', cloud_fraction_Gr)

