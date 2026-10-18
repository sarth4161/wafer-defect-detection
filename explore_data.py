"""
PHASE 1 - Understand the dataset.
Prints: columns, labeled/unlabeled counts, class distribution, wafer map sizes.
Saves : outputs/class_distribution.png, outputs/sample_wafers.png

    python explore_data.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_lswmd, wafer_to_rgb, CLASSES, OUT_DIR

os.makedirs(OUT_DIR, exist_ok=True)
df = load_lswmd()

print("=" * 60)
print("Total wafers in file :", len(df))
print("Unlabeled wafers     :", int(df["label"].isna().sum()), "(no failure label -> not used)")
lab = df.dropna(subset=["label"])
print("Labeled wafers       :", len(lab))

print("\nClass distribution (labeled):")
counts = lab["label"].value_counts()
for c in CLASSES:
    n = int(counts.get(c, 0))
    print(f"  {c:10s} {n:7d}  ({100 * n / len(lab):5.2f}%)")
other = set(counts.index) - set(CLASSES)
if other:
    print("  WARNING unexpected labels:", other)

shapes = np.array([np.asarray(w).shape for w in lab["waferMap"]])
print("\nWafer map height: min %d  median %d  max %d" % (shapes[:, 0].min(), np.median(shapes[:, 0]), shapes[:, 0].max()))
print("Wafer map width : min %d  median %d  max %d" % (shapes[:, 1].min(), np.median(shapes[:, 1]), shapes[:, 1].max()))
print("Unique sizes    :", len({tuple(s) for s in shapes}))
vals = np.unique(np.concatenate([np.unique(np.asarray(w)) for w in lab["waferMap"].iloc[:2000]]))
print("Pixel values    :", vals, "(0 = outside wafer, 1 = good die, 2 = defective die)")
print("Missing waferMap:", int(lab["waferMap"].isna().sum()))
print("=" * 60)

# bar chart
plt.figure(figsize=(8, 4))
vals_ = [int(counts.get(c, 0)) for c in CLASSES]
plt.bar(CLASSES, vals_)
plt.yscale("log")
plt.xticks(rotation=45)
plt.ylabel("number of wafers (log scale)")
plt.title("Class distribution - strongly imbalanced")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=120)
plt.close()

# sample grid: 4 wafers per class
fig, axes = plt.subplots(len(CLASSES), 4, figsize=(8, 2 * len(CLASSES)))
for i, c in enumerate(CLASSES):
    sub = lab[lab["label"] == c]
    for j in range(4):
        ax = axes[i, j]
        ax.axis("off")
        if len(sub) > j:
            ax.imshow(wafer_to_rgb(sub["waferMap"].iloc[j]))
        if j == 0:
            ax.set_title(c, fontsize=9, loc="left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_wafers.png"), dpi=100)
plt.close()
print("Saved plots to outputs/")
