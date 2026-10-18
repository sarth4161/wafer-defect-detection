"""
Saves a few wafer maps per class as PNG images into samples/ so you can try the app.
    python export_samples.py
"""
import os
from utils import load_lswmd, wafer_to_rgb, CLASSES, SAMPLES_DIR

os.makedirs(SAMPLES_DIR, exist_ok=True)
df = load_lswmd().dropna(subset=["label"])
for c in CLASSES:
    sub = df[df["label"] == c].sample(n=min(3, (df["label"] == c).sum()), random_state=1)
    for i, w in enumerate(sub["waferMap"]):
        wafer_to_rgb(w).save(os.path.join(SAMPLES_DIR, f"{c}_{i}.png"))
print("Saved sample images to", SAMPLES_DIR)
