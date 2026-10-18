"""
PHASE 3 - Simple baselines to compare the CNN against.
  1) always predict the most common class
  2) logistic regression on 16x16 downsampled pixels
Results are saved to outputs/baseline_results.json

    python baseline.py
"""
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from utils import NPZ_PATH, OUT_DIR, CLASSES

d = np.load(NPZ_PATH)
X, y, split = d["X"], d["y"], d["split"]
# 64x64 -> 16x16 by averaging 4x4 blocks, then flatten
small = X.astype(np.float32).reshape(len(X), 2, 16, 4, 16, 4).mean(axis=(3, 5)).reshape(len(X), -1) / 255.0
Xtr, ytr, Xte, yte = small[split == 0], y[split == 0], small[split == 2], y[split == 2]

results = {}
major = np.bincount(ytr).argmax()
pred = np.full_like(yte, major)
results["majority_class"] = {"class": CLASSES[major], "accuracy": accuracy_score(yte, pred),
                             "macro_f1": f1_score(yte, pred, average="macro", zero_division=0)}

lr = LogisticRegression(max_iter=300, class_weight="balanced")
lr.fit(Xtr, ytr)
pred = lr.predict(Xte)
results["logistic_regression"] = {"accuracy": accuracy_score(yte, pred),
                                  "macro_f1": f1_score(yte, pred, average="macro", zero_division=0)}

for k, v in results.items():
    print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "baseline_results.json"), "w") as f:
    json.dump(results, f, indent=2)
