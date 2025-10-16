import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # for headless/HPC
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Config
# ======================
FOLDER = "/umbc/rs/nasa-access/users/xingyan/satellite_collocation/satellite_collocation_github/examples/collocate_abi_calipso_local_execution/generate_2017/ABI_CALIOP_collocated_data_with_angles/"
FILE_PREFIX = "ABI_G16_Data_CAL_"
FILE_SUFFIX = ".h5"

CLASS_NAMES_3 = ["Clear (0)", "Water (1)", "Ice (2)"]
CLASS_NAMES_4 = ["Clear (0)", "Water (1)", "Ice (2)", "Unknown (3)"]

SAVE_DIR = "/umbc/rs/nasa-access/xin/cloud-phase-prediction/ABI_Calipso"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# CALIPSO bitfield decode
# ======================
def vfm_feature_flags(val):
    """Decode CALIPSO 16-bit Feature_Classification_Flags (per sample)."""
    val_bit = np.binary_repr(np.uint16(val), width=16)  # pad to 16 bits
    feature_type              = int(val_bit[-3:],      2)
    feature_type_qa           = int(val_bit[-5:-3],    2)
    ice_water_phase           = int(val_bit[-7:-5],    2)
    ice_water_phase_qa        = int(val_bit[-9:-7],    2)
    feature_subtype           = int(val_bit[-12:-9],   2)
    cloud_aerosol_psc_type_qa = int(val_bit[-13],      2)
    horizontal_averaging      = int(val_bit[-16:-13],  2)
    return (feature_type, feature_type_qa, ice_water_phase, ice_water_phase_qa,
            feature_subtype, cloud_aerosol_psc_type_qa, horizontal_averaging)

def Extract_Feature_Info(vfm_array, nlay):
    """
    Decode flags up to nlay[i,0] layers for each profile.
    Returns arrays shaped like vfm_array for the decoded fields.
    """
    npro = nlay.size
    Lmax = vfm_array.shape[1]
    feature_type    = np.full_like(vfm_array, -1)
    feature_type_qa = np.full_like(vfm_array, -1)
    ice_water_phase = np.full_like(vfm_array, -1)
    ice_water_phase_qa = np.full_like(vfm_array, -1)
    feature_subtype = np.full_like(vfm_array, -1)
    cloud_aerosol_psc_type_qa = np.full_like(vfm_array, -1)
    horizontal_averaging = np.full_like(vfm_array, -1)

    for i in range(npro):
        nL = int(nlay[i, 0]) if np.ndim(nlay) == 2 else int(nlay[i])
        nL = max(0, min(nL, Lmax))
        for l in range(nL):
            (ft, ftqa, iwp, iwpqa, fsub, capqa, havg) = vfm_feature_flags(vfm_array[i, l])
            feature_type[i, l] = ft
            feature_type_qa[i, l] = ftqa
            ice_water_phase[i, l] = iwp
            ice_water_phase_qa[i, l] = iwpqa
            feature_subtype[i, l] = fsub
            cloud_aerosol_psc_type_qa[i, l] = capqa
            horizontal_averaging[i, l] = havg

    return (feature_type, feature_type_qa, ice_water_phase, ice_water_phase_qa,
            feature_subtype, cloud_aerosol_psc_type_qa, horizontal_averaging)

# ======================
# Label mappings → 0/1/2/3 (3=Unknown)
# ======================
def map_calipso_to_4class(ice_water_phase_top):
    """
    CALIPSO top-layer mapping:
      (1 or 3) → Ice(2), 2 → Water(1), 65535 → Clear(0), 0 → Unknown(3)
      everything else → Unknown(3)
    """
    x = np.asarray(ice_water_phase_top).astype(int).ravel()
    out = np.full_like(x, 3, dtype=int)       # default Unknown(3)
    out[(x == 1) | (x == 3)] = 2              # Ice
    out[x == 2]              = 1              # Water
    out[x == 65535]          = 0              # Clear
    out[x == 0]              = 3              # Unknown (explicit per request)
    return out

def map_abi_to_4class(abi_cloud_phase):
    """
    ABI mapping:
      0 → Clear(0), 1/2 → Water(1), 4 → Ice(2),
      ANY OTHER (e.g., 3 Mixed, 5 Unknown, etc.) → Unknown(3)
    """
    v = np.asarray(abi_cloud_phase).astype(int).ravel()
    out = np.full_like(v, 3, dtype=int)       # default Unknown(3)
    out[v == 0] = 0
    out[(v == 1) | (v == 2)] = 1
    out[v == 4] = 2
    # All others remain 3 (Unknown)
    return out

# ======================
# Utilities
# ======================
def counts_and_pct(a, K):
    cnt = np.bincount(a, minlength=K)[:K].astype(int)
    tot = int(cnt.sum())
    pct = np.where(tot > 0, cnt / max(tot, 1) * 100.0, 0.0)
    return cnt, pct, tot

# ======================
# Scan all files and collect labels
# ======================
cal_all = []
abi_all = []
n_files = 0

for fname in os.listdir(FOLDER):
    if not (fname.startswith(FILE_PREFIX) and fname.endswith(FILE_SUFFIX)):
        continue
    fpath = os.path.join(FOLDER, fname)
    try:
        ds = xr.open_dataset(fpath)

        # --- CALIPSO (decode flags → top-layer ice/water phase → 4-class)
        vfm = ds["CALIOP_Clay_Feature_Classification_Flags_1km"].values
        nly = ds["CALIOP_N_Clay_1km"].values
        _, _, ice_water_phase, _, _, _, _ = Extract_Feature_Info(vfm, nly)
        cal_top = ice_water_phase[:, 0]                     # top-most layer
        cal4 = map_calipso_to_4class(cal_top)

        # --- ABI (direct map to 4-class)
        abi_phase = ds["ABI_Cloud_Phase"].values
        abi4 = map_abi_to_4class(abi_phase)

        # --- align by min length
        N = min(cal4.size, abi4.size)
        cal4 = cal4[:N]
        abi4 = abi4[:N]

        cal_all.append(cal4)
        abi_all.append(abi4)

        n_files += 1
        ds.close()
    except Exception as e:
        print(f"[WARN] Skipping {fname}: {e}")

if not cal_all:
    raise RuntimeError("No labels collected. Check file paths/variables.")

cal_all = np.concatenate(cal_all)
abi_all = np.concatenate(abi_all)

print(f"\nProcessed files: {n_files}")
print(f"Total aligned pairs (incl. Unknown): {cal_all.size:,}")

# ======================
# Print distributions including Unknown (0/1/2/3)
# ======================
K4 = 4
cal_cnt4, cal_pct4, cal_tot4 = counts_and_pct(cal_all, K4)
abi_cnt4, abi_pct4, abi_tot4 = counts_and_pct(abi_all, K4)

print("\nCALIPSO distribution (all aligned pairs, incl. Unknown):")
for i, name in enumerate(CLASS_NAMES_4):
    print(f"  {name:<12}: {cal_cnt4[i]:,} ({cal_pct4[i]:.1f}%)")

print("\nABI distribution (all aligned pairs, incl. Unknown):")
for i, name in enumerate(CLASS_NAMES_4):
    print(f"  {name:<12}: {abi_cnt4[i]:,} ({abi_pct4[i]:.1f}%)")

# ======================
# Restrict to valid 3-class pairs (exclude Unknown=3)
# ======================
valid3 = (cal_all < 3) & (abi_all < 3)  # both in {0,1,2}
cal_3 = cal_all[valid3]
abi_3 = abi_all[valid3]
print(f"\nValid matched pairs used in plots (exclude Unknown): {cal_3.size:,}")
print(f"Exact agreement over valid pairs: {np.mean(cal_3 == abi_3) * 100:.2f}%")

# ======================
# Confusion matrix for 3 classes (Y=CAL, X=ABI)
# ======================
K3 = 3
cm = np.zeros((K3, K3), dtype=int)
for y, x in zip(cal_3, abi_3):
    cm[y, x] += 1

row_sums = cm.sum(axis=1, keepdims=True).astype(float)
with np.errstate(divide="ignore", invalid="ignore"):
    cm_rowpct = np.where(row_sums > 0, cm / row_sums * 100.0, 0.0)

annot = np.empty_like(cm, dtype=object)
for i in range(K3):
    for j in range(K3):
        annot[i, j] = f"{cm[i, j]:,}\n({cm_rowpct[i, j]:.1f}%)"

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    cm,
    annot=annot, fmt="",
    cmap="Blues",
    cbar=True,
    xticklabels=[n.split()[0] for n in CLASS_NAMES_3],  # Clear Water Ice
    yticklabels=[n.split()[0] for n in CLASS_NAMES_3],
    linewidths=0.5, linecolor="white"
)
ax.set_xlabel("ABI Prediction")
ax.set_ylabel("CALIPSO Label")
ax.set_title("Confusion Matrix (Valid Pairs): Count and Row-Percentage")
plt.tight_layout()
cm_path = os.path.join(SAVE_DIR, "confusion_matrix_water_iceNewLabelV2New.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {cm_path}")

# ======================
# Histogram (3 classes only) — Unknown not plotted
# ======================
def counts_only_3(a):
    cnt = np.bincount(a, minlength=3)[:3].astype(int)
    tot = int(cnt.sum())
    pct = np.where(tot > 0, cnt / tot * 100.0, 0.0)
    return cnt, pct, tot

cal_cnt3, cal_pct3, cal_tot3 = counts_only_3(cal_3)
abi_cnt3, abi_pct3, abi_tot3 = counts_only_3(abi_3)

x = np.arange(3)
w = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
bars_cal = ax.bar(x - w/2, cal_cnt3, width=w, label=f"CALIPSO (n={cal_tot3:,})")
bars_abi = ax.bar(x + w/2, abi_cnt3, width=w, label=f"ABI (n={abi_tot3:,})")

for i in range(3):
    ax.text(x[i] - w/2, cal_cnt3[i] + max(1, 0.01*cal_tot3), f"{cal_pct3[i]:.1f}%",
            ha="center", va="bottom", fontsize=9)
    ax.text(x[i] + w/2, abi_cnt3[i] + max(1, 0.01*abi_tot3), f"{abi_pct3[i]:.1f}%",
            ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(["Clear", "Water", "Ice"])
ax.set_ylabel("Count")
ax.set_title("Label Distribution")
ax.legend()
ax.margins(y=0.1)
plt.tight_layout()
hist_path = os.path.join(SAVE_DIR, "water_ice_histogramNewLabelV2New.png")
plt.savefig(hist_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {hist_path}")

