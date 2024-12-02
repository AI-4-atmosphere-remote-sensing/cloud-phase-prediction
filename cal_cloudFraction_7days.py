import numpy as np
from netCDF4 import Dataset
import os

# Function to calculate the solar geometry
def calculate_solar_geometry_v03(solar_zenith, solar_azimuth):
    solar_zenith_rad = np.deg2rad(solar_zenith)
    solar_azimuth_rad = np.deg2rad(solar_azimuth)
    return solar_zenith_rad, solar_azimuth_rad

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
    base_name = os.path.basename(v03_filename).replace('.nc', '_prediction_onlyM16.nc')
    return os.path.join(prediction_folder, base_name)

# Base paths for the folders
v03_base_folder = '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/VNP03MOD/2017/'
prediction_base_folder = '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/predict_2016/2017/'

# List of folder names for the first seven days
days = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016','017','018','019','020','021','022','023','024','025','026','027','028','029','030','031']

# Initialize grids for counting total pixels and cloud pixels across all days
grid_size = (180, 360)
total_pixels_agg = np.zeros(grid_size, dtype=int)
liquid_cloud_pixels_agg = np.zeros(grid_size, dtype=int)
ice_cloud_pixels_agg = np.zeros(grid_size, dtype=int)

# Define the x-axis range to be used
x_min, x_max = 1100, 2100

# Iterate over the first seven days
for day in days:
    # Set paths for the current day
    v03_folder = os.path.join(v03_base_folder, day)
    prediction_folder = os.path.join(prediction_base_folder, day)

    i = 1
    # Iterate over each file in the current day's VNP03 folder
    for v03_file in os.listdir(v03_folder):
        print(f"Day {day}, File {i}")
        i += 1
        if v03_file.endswith('.nc'):
            file_v03 = os.path.join(v03_folder, v03_file)
            file_prediction = find_matching_prediction_file(v03_file, prediction_folder)

            # Read latitude, longitude, prediction class data, and sensor_zenith data
            with Dataset(file_v03, 'r') as nc:
                lon = nc['/geolocation_data/longitude'][:, x_min:x_max]
                lat = nc['/geolocation_data/latitude'][:, x_min:x_max]
                solar_zenith = nc['/geolocation_data/solar_zenith'][:, x_min:x_max]
                solar_azimuth = nc['/geolocation_data/solar_azimuth'][:, x_min:x_max]
                view_zenith = nc['/geolocation_data/sensor_zenith'][:, x_min:x_max]
                view_azimuth = nc['/geolocation_data/sensor_azimuth'][:, x_min:x_max]

            with Dataset(file_prediction, 'r') as nc:
                prediction = nc.variables['prediction'][:, x_min:x_max]

            # Calculate the glint angle and apply glint filter
            glint_angle = calculate_glint_angle(solar_zenith, solar_azimuth, view_zenith, view_azimuth)
            glint_filter = apply_glint_filter(glint_angle)

            # Apply solar zenith filter (i.e., solar_zenith <= 83)
            valid_pixels = (solar_zenith <= 83) & glint_filter
            lat = lat[valid_pixels]
            lon = lon[valid_pixels]
            prediction = prediction[valid_pixels]

            # Convert latitude and longitude to grid indices
            grid_lat = np.clip(np.round(lat).astype(int) + 90, 0, 179)
            grid_lon = np.clip(np.round(lon).astype(int) + 180, 0, 359)

            # Update counts using vectorized operations for the current file
            np.add.at(total_pixels_agg, (grid_lat.flatten(), grid_lon.flatten()), 1)
            np.add.at(liquid_cloud_pixels_agg, (grid_lat[prediction == 1].flatten(), grid_lon[prediction == 1].flatten()), 1)
            np.add.at(ice_cloud_pixels_agg, (grid_lat[prediction == 2].flatten(), grid_lon[prediction == 2].flatten()), 1)

# Calculate aggregate cloud fraction for liquid and ice clouds together after processing all files
cloud_fraction_agg = np.full(grid_size, np.nan, dtype=float)  # Initialize with NaN

# Handle divisions by zero or missing data
valid_counts = total_pixels_agg > 0
cloud_fraction_agg[valid_counts] = (liquid_cloud_pixels_agg[valid_counts] + ice_cloud_pixels_agg[valid_counts]) / total_pixels_agg[valid_counts]

# Save the aggregated cloud_fraction to a file
np.save('/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/cloud_fraction2017_31days.npy', cloud_fraction_agg)

