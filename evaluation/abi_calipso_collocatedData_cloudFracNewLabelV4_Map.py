import os, glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
INPUT_GLOB = '/umbc/rs/nasa-access/users/xingyan/satellite_collocation/satellite_collocation_github/examples/collocate_abi_calipso_local_execution/generate_2017/ABI_CALIOP_collocated_data_with_angles/*.h5'
OUT_DIR = '/umbc/rs/nasa-access/xin/cloud-phase-prediction/ABI_Calipso'
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Cartopy cache to avoid ~/.local space issues ----
CARTOPY_CACHE = os.path.join(OUT_DIR, "_cartopy_data")
os.environ["CARTOPY_DATA_DIR"] = CARTOPY_CACHE
os.makedirs(CARTOPY_CACHE, exist_ok=True)

# Try Cartopy; fall back gracefully
_HAS_CARTOPY = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy
    cartopy.config["data_dir"] = CARTOPY_CACHE
    _HAS_CARTOPY = True
except Exception as _e:
    print(f"[info] Cartopy unavailable ({_e}); using rectangular fallback (no outlines).")

# ============================================================
# CALIPSO bit decode
# ============================================================
def vfm_feature_flags(val):
    b = np.binary_repr(np.uint16(val), width=16)
    feature_type              = int(b[-3:],      2)
    feature_type_qa           = int(b[-5:-3],    2)
    ice_water_phase           = int(b[-7:-5],    2)
    ice_water_phase_qa        = int(b[-9:-7],    2)
    feature_subtype           = int(b[-12:-9],   2)
    cloud_aerosol_psc_type_qa = int(b[-13],      2)
    horizontal_averaging      = int(b[-16:-13],  2)
    return (feature_type, feature_type_qa, ice_water_phase, ice_water_phase_qa,
            feature_subtype, cloud_aerosol_psc_type_qa, horizontal_averaging)

def Extract_Feature_Info(vfm_array, nlay):
    npro = nlay.size
    feature_type    = np.full_like(vfm_array, -1, dtype=int)
    feature_type_qa = np.full_like(vfm_array, -1, dtype=int)
    ice_water_phase = np.full_like(vfm_array, -1, dtype=int)
    ice_water_phase_qa = np.full_like(vfm_array, -1, dtype=int)
    feature_subtype = np.full_like(vfm_array, -1, dtype=int)
    cloud_aerosol_psc_type_qa = np.full_like(vfm_array, -1, dtype=int)
    horizontal_averaging = np.full_like(vfm_array, -1, dtype=int)
    for i in range(npro):
        for l in range(int(nlay[i, 0])):
            ft, ftqa, iwp, iwpqa, fsub, capqa, havg = vfm_feature_flags(vfm_array[i, l])
            feature_type[i, l]    = ft
            feature_type_qa[i, l] = ftqa
            ice_water_phase[i, l] = iwp
            ice_water_phase_qa[i, l] = iwpqa
            feature_subtype[i, l] = fsub
            cloud_aerosol_psc_type_qa[i, l] = capqa
            horizontal_averaging[i, l] = havg
    return (feature_type, feature_type_qa, ice_water_phase, ice_water_phase_qa,
            feature_subtype, cloud_aerosol_psc_type_qa, horizontal_averaging)

# ============================================================
# 1°×1° grid helpers
# ============================================================
GRID_SHAPE = (180, 360)  # (lat bins, lon bins) for [-90..90), [-180..180)

def latlon_to_grid(lat, lon):
    ilat = np.clip(np.floor(lat).astype(int) + 90,  0, 179)
    ilon = np.clip(np.floor(lon).astype(int) + 180, 0, 359)
    return ilat, ilon

# Grid cell centers for computing plot extent
LAT_CENTERS = np.arange(-90, 90, 1.0) + 0.5   # 180
LON_CENTERS = np.arange(-180, 180, 1.0) + 0.5 # 360

def data_extent_from_cf(cf, pad_deg=2):
    """Compute lon/lat extent covering finite data, with padding."""
    rr, cc = np.where(np.isfinite(cf))
    if rr.size == 0:
        return [-180, 180, -90, 90]
    lat_min = LAT_CENTERS[rr.min()] - 0.5 - pad_deg
    lat_max = LAT_CENTERS[rr.max()] + 0.5 + pad_deg
    lon_min = LON_CENTERS[cc.min()] - 0.5 - pad_deg
    lon_max = LON_CENTERS[cc.max()] + 0.5 + pad_deg
    lat_min = max(-90, lat_min); lat_max = min(90, lat_max)
    lon_min = max(-180, lon_min); lon_max = min(180, lon_max)
    return [lon_min, lon_max, lat_min, lat_max]

# ============================================================
# Accumulators
# ============================================================
total_cal = np.zeros(GRID_SHAPE, dtype=int)
cloud_cal = np.zeros(GRID_SHAPE, dtype=int)  # (Ice + Water)
total_abi = np.zeros(GRID_SHAPE, dtype=int)
cloud_abi = np.zeros(GRID_SHAPE, dtype=int)  # (labels 1,2,3,4,5)

cal_counts = dict(Clear=0, Water=0, Ice=0, Other=0)
abi_counts = dict(Clear=0, Water=0, Ice=0, Other=0)

# ============================================================
# Iterate files & accumulate
# ============================================================
files = sorted(glob.glob(INPUT_GLOB))
print(f"Found {len(files)} files")

for f in files:
    try:
        ds = xr.open_dataset(f)

        # CALIPSO geolocation
        lon_cal = np.asarray(ds['CALIPSO_Lon']).ravel()
        lat_cal = np.asarray(ds['CALIPSO_Lat']).ravel()

        # CALIPSO VFM -> top-layer phase -> remap
        vfm = ds['CALIOP_Clay_Feature_Classification_Flags_1km'].values
        nly = ds['CALIOP_N_Clay_1km'].values
        (_, _, iwp, _, _, _, _) = Extract_Feature_Info(vfm, nly)
        top = iwp[:, 0].astype(np.int64)

        cal_new = np.full_like(top, -1, dtype=np.int64)
        cal_new[(top == 1) | (top == 3)] = 2   # Ice
        cal_new[top == 2] = 1                  # Water
        cal_new[top == 65535] = 0              # Clear (no layer -> wrapped)
        cal_new[top == 0] = 3                  # Unknown

        # ABI geolocation
        lon_abi = np.asarray(ds['ABI_Lon_1km']).ravel()
        lat_abi = np.asarray(ds['ABI_Lat_1km']).ravel()

        # ABI cloud phase
        abi = np.asarray(ds['ABI_Cloud_Phase']).ravel().astype(float)

        ds.close()

        # ---- CALIPSO ----
        M_cal = min(lon_cal.size, lat_cal.size, cal_new.size)
        lon_c, lat_c, cal_c = lon_cal[:M_cal], lat_cal[:M_cal], cal_new[:M_cal]
        mask_c = np.isfinite(lon_c) & np.isfinite(lat_c) & np.isfinite(cal_c)
        if np.any(mask_c):
            lon_c = lon_c[mask_c]; lat_c = lat_c[mask_c]; cal_c = cal_c[mask_c]
            ilat_c, ilon_c = latlon_to_grid(lat_c, lon_c)

            is_cal_clear = (cal_c == 0)
            is_cal_water = (cal_c == 1)
            is_cal_ice   = (cal_c == 2)
            is_cal_other = ~(is_cal_clear | is_cal_water | is_cal_ice)

            np.add.at(total_cal, (ilat_c, ilon_c), 1)
            np.add.at(cloud_cal, (ilat_c[is_cal_water | is_cal_ice],
                                  ilon_c[is_cal_water | is_cal_ice]), 1)

            cal_counts['Clear'] += int(np.sum(is_cal_clear))
            cal_counts['Water'] += int(np.sum(is_cal_water))
            cal_counts['Ice']   += int(np.sum(is_cal_ice))
            cal_counts['Other'] += int(np.sum(is_cal_other))

        # ---- ABI ----
        M_abi = min(lon_abi.size, lat_abi.size, abi.size)
        lon_a, lat_a, abi_a = lon_abi[:M_abi], lat_abi[:M_abi], abi[:M_abi]
        mask_a = np.isfinite(lon_a) & np.isfinite(lat_a) & np.isfinite(abi_a)
        if np.any(mask_a):
            lon_a = lon_a[mask_a]; lat_a = lat_a[mask_a]; abi_a = abi_a[mask_a]
            ilat_a, ilon_a = latlon_to_grid(lat_a, lon_a)

            is_abi_clear = (abi_a == 0)
            is_abi_water = (abi_a == 1) | (abi_a == 2)
            is_abi_ice   = (abi_a == 4)
            is_abi_other = ~(is_abi_clear | is_abi_water | is_abi_ice)  # includes 3,5

            np.add.at(total_abi, (ilat_a, ilon_a), 1)
            # Cloud = any of 1,2,3,4,5
            is_abi_cloud = (abi_a != 0)
            np.add.at(cloud_abi, (ilat_a[is_abi_cloud], ilon_a[is_abi_cloud]), 1)

            abi_counts['Clear'] += int(np.sum(is_abi_clear))
            abi_counts['Water'] += int(np.sum(is_abi_water))
            abi_counts['Ice']   += int(np.sum(is_abi_ice))
            abi_counts['Other'] += int(np.sum(is_abi_other))

    except Exception as e:
        print(f"[error] {f}: {e}")

# ============================================================
# Cloud fractions
# ============================================================
cloud_frac_cal = np.full(GRID_SHAPE, np.nan, dtype=float)
cloud_frac_abi = np.full(GRID_SHAPE, np.nan, dtype=float)
mask_cal = total_cal > 0
mask_abi = total_abi > 0
cloud_frac_cal[mask_cal] = cloud_cal[mask_cal] / total_cal[mask_cal]
cloud_frac_abi[mask_abi] = cloud_abi[mask_abi] / total_abi[mask_abi]

print("\n=== Global Label Counts (for reference) ===")
print(f"CALIPSO -> Clear: {cal_counts['Clear']:,} | Water: {cal_counts['Water']:,} | Ice: {cal_counts['Ice']:,} | Other: {cal_counts['Other']:,}")
print(f"ABI     -> Clear: {abi_counts['Clear']:,} | Water: {abi_counts['Water']:,} | Ice: {abi_counts['Ice']:,} | Other: {abi_counts['Other']:,}")

# ============================================================
# Plot: SAME extent for both (use ABI extent)
# ============================================================
def _rectangular_fallback(cf, title, out_path, extent):
    plt.figure(figsize=(10, 8))  # same size for both
    cmap = plt.cm.viridis.copy(); cmap.set_bad(alpha=0)
    data = np.ma.masked_invalid(cf)
    im = plt.imshow(data, origin='lower', extent=[-180, 180, -90, 90],
                    vmin=0, vmax=1, interpolation='nearest', cmap=cmap)
    plt.xlim(extent[0], extent[1]); plt.ylim(extent[2], extent[3])
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Cloud Fraction')
    plt.title(title)
    plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.tight_layout(); plt.savefig(out_path, dpi=220); plt.close()
    print(f"[saved fallback] {out_path}")

def plot_map_same_extent(cf, title, out_path, extent):
    if _HAS_CARTOPY:
        try:
            fig = plt.figure(figsize=(10, 8))  # same size for both
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())

            # Basemap underlay (identical settings)
            ax.add_feature(cfeature.OCEAN, facecolor='0.88', zorder=0)
            ax.add_feature(cfeature.LAND,  facecolor='0.93', zorder=0)
            ax.coastlines('110m', linewidth=0.8, zorder=2)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=2)

            cmap = plt.cm.viridis.copy(); cmap.set_bad(alpha=0)
            data = np.ma.masked_invalid(cf)
            im = ax.imshow(
                data, origin='lower', extent=[-180, 180, -90, 90],
                transform=ccrs.PlateCarree(), vmin=0, vmax=1,
                interpolation='nearest', cmap=cmap, zorder=1
            )

            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle='--')
            gl.right_labels = False; gl.top_labels = False

            cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
            cbar.set_label('Cloud Fraction')

            ax.set_title(title, fontsize=14, pad=10)
            plt.tight_layout(); plt.savefig(out_path, dpi=220); plt.close(fig)
            print(f"[saved] {out_path}")
            return
        except Exception as e:
            print(f"[warn] Cartopy plotting failed ({e}); using fallback.")

    _rectangular_fallback(cf, title, out_path, extent)

# --- Compute ONE extent from ABI and use it for both maps ---
abi_extent = data_extent_from_cf(cloud_frac_abi, pad_deg=2)

plot_map_same_extent(
    cloud_frac_abi,
    "ABI 1°×1° Cloud Fraction (Liquid/Supercooled/Mixed/Ice/Unknown)",
    os.path.join(OUT_DIR, "CloudFraction_ABI_Map.png"),
    abi_extent,
)

plot_map_same_extent(
    cloud_frac_cal,
    "CALIPSO 1°×1° Cloud Fraction (Ice + Water)",
    os.path.join(OUT_DIR, "CloudFraction_CALIPSO_Map.png"),
    abi_extent,   # <<< same extent as ABI
)

