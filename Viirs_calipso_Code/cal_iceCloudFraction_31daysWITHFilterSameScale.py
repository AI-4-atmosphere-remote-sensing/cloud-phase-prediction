import numpy as np
from netCDF4 import Dataset
import os

def calculate_glint_angle(solar_zenith, solar_azimuth, view_zenith, view_azimuth):
    phi = np.deg2rad(solar_azimuth - view_azimuth)
    cos_glint = (np.cos(np.deg2rad(view_zenith)) * np.cos(np.deg2rad(solar_zenith)) -
                 np.sin(np.deg2rad(view_zenith)) * np.sin(np.deg2rad(solar_zenith)) * np.cos(phi))
    return np.degrees(np.arccos(np.clip(cos_glint, -1.0, 1.0)))

def apply_glint_filter(glint_angle, threshold=40):
    return glint_angle > threshold

def find_matching_prediction_file_pred(v03_filename, prediction_folder):
    base_name = os.path.basename(v03_filename).replace('.nc', '_prediction_onlyM16.nc')
    return os.path.join(prediction_folder, base_name)

def find_matching_prediction_file_GT(v03_filename, prediction_folder):
    parts = v03_filename.split('.')
    date_part = parts[0].split('_')[1]
    time_part = parts[1]
    prediction_filename = f'CLDPROP_L2_VIIRS_SNPP.{date_part}.{time_part}.nc'
    return os.path.join(prediction_folder, prediction_filename)

# === Paths and setup ===
v03_base = '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/VNP03MOD/2017/'
pred_base = '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/predict_2016/2017/'
gt_base = '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/ctphase/CLDPROP_L2_VIIRS_SNPP/2017/'
days = ['{:03d}'.format(d) for d in range(1, 32)]
grid_shape = (180, 360)
x_min, x_max = 1100, 2100

# === Accumulators ===
ice_pred = np.zeros(grid_shape, dtype=int)
cloud_pred = np.zeros(grid_shape, dtype=int)
ice_gt = np.zeros(grid_shape, dtype=int)
cloud_gt = np.zeros(grid_shape, dtype=int)

for day in days:
    print(f"Processing day {day}")
    v03_folder = os.path.join(v03_base, day)
    pred_folder = os.path.join(pred_base, day)
    gt_folder = os.path.join(gt_base, day)

    for i, fname in enumerate(os.listdir(v03_folder), 1):
        if not fname.endswith('.nc'):
            continue
        print(f" Day {day}, File {i}: {fname}")
        v03_file = os.path.join(v03_folder, fname)
        pred_file = find_matching_prediction_file_pred(fname, pred_folder)
        gt_file = find_matching_prediction_file_GT(fname, gt_folder)

        if not os.path.exists(pred_file):
            print(" Missing prediction:", pred_file)
            continue
        if not os.path.exists(gt_file):
            print(" Missing GT:", gt_file)
            continue

        with Dataset(v03_file) as nc:
            lon = nc['/geolocation_data/longitude'][:, x_min:x_max]
            lat = nc['/geolocation_data/latitude'][:, x_min:x_max]
            sza = nc['/geolocation_data/solar_zenith'][:, x_min:x_max]
            saa = nc['/geolocation_data/solar_azimuth'][:, x_min:x_max]
            vza = nc['/geolocation_data/sensor_zenith'][:, x_min:x_max]
            vaa = nc['/geolocation_data/sensor_azimuth'][:, x_min:x_max]

        with Dataset(pred_file) as nc:
            pred = nc.variables['prediction'][:, x_min:x_max]

        with Dataset(gt_file) as nc:
            if 'Cloud_Phase_Cloud_Top_Properties' in nc.groups['geophysical_data'].variables:
                gt = nc.groups['geophysical_data'].variables['Cloud_Phase_Cloud_Top_Properties'][:, x_min:x_max]
            else:
                print(" Missing GT phase data in:", gt_file)
                continue

        glint = calculate_glint_angle(sza, saa, vza, vaa)
        mask = (sza <= 83) & apply_glint_filter(glint)
        if not np.any(mask):
            print(" No valid pixels after filters.")
            continue

        lat_valid = lat[mask]
        lon_valid = lon[mask]
        pred_valid = pred[mask]
        gt_valid = gt[mask]

        glat = np.clip(np.floor(lat_valid).astype(int) + 90, 0, 179)
        glon = np.clip(np.floor(lon_valid).astype(int) + 180, 0, 359)

        # Pred
        pred_ice = pred_valid == 2
        pred_cloud = (pred_valid == 1) | pred_ice
        np.add.at(ice_pred, (glat[pred_ice], glon[pred_ice]), 1)
        np.add.at(cloud_pred, (glat[pred_cloud], glon[pred_cloud]), 1)

        # GT
        gt_ice = gt_valid == 2
        gt_cloud = (gt_valid == 1) | gt_ice
        np.add.at(ice_gt, (glat[gt_ice], glon[gt_ice]), 1)
        np.add.at(cloud_gt, (glat[gt_cloud], glon[gt_cloud]), 1)

# === Compute ice fractions ===
ice_fraction_pred = np.full(grid_shape, np.nan)
ice_fraction_gt = np.full(grid_shape, np.nan)

valid_pred = cloud_pred > 0
valid_gt = cloud_gt > 0

ice_fraction_pred[valid_pred] = ice_pred[valid_pred] / cloud_pred[valid_pred]
ice_fraction_gt[valid_gt] = ice_gt[valid_gt] / cloud_gt[valid_gt]

# === Save ===
np.save('/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/ice_fraction2017_31days.npy', ice_fraction_pred)
np.save('/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/ice_fraction_GT2017_31days.npy', ice_fraction_gt)

print("Ice cloud fraction processing complete.")

