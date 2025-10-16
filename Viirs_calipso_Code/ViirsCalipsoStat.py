import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless save on HPC
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Paths & basic config
# =========================
NPZ_PATH = "/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/training/2016.npz"
OUT_DIR  = "/umbc/rs/nasa-access/xin/cloud-phase-prediction"
TAG = "2016"  # used in filenames
CLASS_NAMES = ["Clear", "Water", "Ice"]  # assumes classes 0,1,2 in this order
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Helpers
# =========================
def normalize_labels(a):
    """Map {1,2,3} -> {0,1,2} if needed, return int array (raveled)."""
    a = np.asarray(a).ravel()
    u = np.unique(a)
    if set(u.tolist()) == {1, 2, 3}:
        a = a - 1
    return a.astype(int)

def counts_and_perc(a, K):
    """Counts and percentages for labels in 0..K-1."""
    cnt = np.bincount(a, minlength=K)[:K].astype(int)
    tot = int(cnt.sum())
    pct = np.where(tot > 0, cnt / tot * 100.0, 0.0)
    return cnt, pct, tot

def print_dist(name, cnt, pct):
    print(f"\n{name} label distribution (valid pairs):")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:>5}: {cnt[i]:,} ({pct[i]:.1f}%)")

# =========================
# Load labels
# =========================
data = np.load(NPZ_PATH)
labels = data["label"]            # [:,0]=CALIPSO, [:,1]=VIIRS (per your code)
cal = normalize_labels(labels[:, 0])
vir = normalize_labels(labels[:, 1])

# keep only valid pairs (0..K-1 on both sides)
K = len(CLASS_NAMES)
valid = np.isin(cal, np.arange(K)) & np.isin(vir, np.arange(K))
cal = cal[valid]
vir = vir[valid]

print(f"Valid matched pairs: {cal.size:,}")
print(f"Exact agreement rate: {np.mean(cal == vir) * 100:.2f}%")

# =========================
# Confusion matrix (Y=CAL row, X=VIIRS col)
# =========================
cm = np.zeros((K, K), dtype=np.int64)
for y, x in zip(cal, vir):
    cm[y, x] += 1

row_sums = cm.sum(axis=1, keepdims=True).astype(float)
with np.errstate(divide="ignore", invalid="ignore"):
    cm_rowpct = np.where(row_sums > 0, cm / row_sums * 100.0, 0.0)

# cell annotations: "count\n(percentage%)"
annot = np.empty_like(cm, dtype=object)
for i in range(K):
    for j in range(K):
        annot[i, j] = f"{cm[i, j]:,}\n({cm_rowpct[i, j]:.1f}%)"

# =========================
# Plot & save: Confusion matrix
# =========================
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    cm,
    annot=annot, fmt="",
    cmap="Blues",
    cbar=True,
    xticklabels=[f"{n} (VIIRS)" for n in CLASS_NAMES],
    yticklabels=[f"{n} (CAL)"   for n in CLASS_NAMES],
    linewidths=0.5, linecolor="white"
)
ax.set_xlabel("VIIRS Prediction")
ax.set_ylabel("CALIPSO Label")
ax.set_title("Confusion Matrix (Valid Matched Pairs): Count and Row-Percentage")
plt.tight_layout()
cm_png = os.path.join(OUT_DIR, f"confusion_matrix_Viirs_calipsoLabels.png")
plt.savefig(cm_png, dpi=300, bbox_inches="tight")
plt.close()

# =========================
# Distributions per sensor
# =========================
cal_cnt, cal_pct, cal_tot = counts_and_perc(cal, K)
vir_cnt, vir_pct, vir_tot = counts_and_perc(vir, K)

print_dist("CALIPSO", cal_cnt, cal_pct)
print_dist("VIIRS",   vir_cnt, vir_pct)

# =========================
# Plot & save: Histogram (same plot, both sensors)
# =========================
x = np.arange(K)
w = 0.38

fig, ax = plt.subplots(figsize=(8, 5))
bars_cal = ax.bar(x - w/2, cal_cnt, width=w, label=f"CALIPSO (n={cal_tot:,})")
bars_vir = ax.bar(x + w/2, vir_cnt, width=w, label=f"VIIRS (n={vir_tot:,})")

# annotate percentages above bars
for i in range(K):
    ax.text(x[i] - w/2, cal_cnt[i] + max(1, 0.01*cal_tot),
            f"{cal_pct[i]:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.text(x[i] + w/2, vir_cnt[i] + max(1, 0.01*vir_tot),
            f"{vir_pct[i]:.1f}%", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES)
ax.set_ylabel("Count")
ax.set_title("Label Distribution (Valid Pairs) Calipso and Viirs")
ax.legend()
ax.margins(y=0.1)
plt.tight_layout()
hist_png = os.path.join(OUT_DIR, f"label_hist__Viirs_calipsoLabels.png")
plt.savefig(hist_png, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved: {cm_png}")
print(f"Saved: {hist_png}")

