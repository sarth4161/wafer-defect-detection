"""
Creates a FAKE dataset in the same format as WM-811K (data/LSWMD.pkl).
Only for testing that the code runs when you do not have the real dataset yet.
Results on fake data mean nothing - use the real WM-811K for the project.

    python make_fake_data.py
"""
import os
import numpy as np
import pandas as pd
from utils import PKL_PATH, DATA_DIR, CLASSES, set_seed

rng = np.random.default_rng(0)


def base_wafer(n):
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) / (n / 2)   # 0 centre .. 1 edge
    ang = np.arctan2(yy - c, xx - c)
    w = np.where(r <= 1.0, 1, 0).astype(np.uint8)
    return w, r, yy, xx, ang


def make_wafer(label):
    n = int(rng.integers(26, 70))
    w, r, yy, xx, ang = base_wafer(n)
    inside = w == 1
    d = np.zeros_like(inside)
    if label == "Center":
        d = r < rng.uniform(0.2, 0.35)
    elif label == "Donut":
        r0 = rng.uniform(0.4, 0.6)
        d = np.abs(r - r0) < 0.12
    elif label == "Edge-Loc":
        a0 = rng.uniform(-np.pi, np.pi)
        d = (r > 0.7) & (np.abs(np.angle(np.exp(1j * (ang - a0)))) < rng.uniform(0.3, 0.5))
    elif label == "Edge-Ring":
        d = r > rng.uniform(0.82, 0.9)
    elif label == "Loc":
        cy, cx = rng.uniform(0.25, 0.75, 2) * n
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < n * rng.uniform(0.08, 0.14)
    elif label == "Random":
        d = rng.random((n, n)) < rng.uniform(0.1, 0.25)
    elif label == "Scratch":
        a = rng.uniform(0, np.pi)
        dist = np.abs((xx - n / 2) * np.sin(a) - (yy - n / 2) * np.cos(a))
        d = dist < 1.2
    elif label == "Near-full":
        d = rng.random((n, n)) < rng.uniform(0.75, 0.9)
    # tiny background noise on every wafer
    d = d | (rng.random((n, n)) < 0.01)
    w = w.copy()
    w[inside & d] = 2
    return w


if __name__ == "__main__":
    set_seed(0)
    counts = {"Center": 300, "Donut": 120, "Edge-Loc": 300, "Edge-Ring": 300, "Loc": 250,
              "Random": 150, "Scratch": 100, "Near-full": 50, "none": 1500}
    rows = []
    for lab, k in counts.items():
        for _ in range(k):
            fail = np.array([[lab]], dtype=object)
            rows.append((make_wafer(lab if lab != "none" else "x"), fail))
    for _ in range(300):  # unlabeled wafers, like in the real dataset
        rows.append((make_wafer("Random"), np.array([], dtype=object).reshape(0, 0)))
    order = rng.permutation(len(rows))
    n = len(order)
    df = pd.DataFrame({"waferMap": [rows[i][0] for i in order],
                       "failureType": [rows[i][1] for i in order],
                       # like the real dataset: 25 wafers per lot, lot names lot1, lot2, ...
                       "lotName": [np.array(["lot%d" % (k // 25 + 1)]) for k in range(n)],
                       "waferIndex": [float(k % 25 + 1) for k in range(n)]})
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_pickle(PKL_PATH)
    print(f"Saved fake dataset with {len(df)} wafers -> {PKL_PATH}")
