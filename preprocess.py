"""
PHASE 1 - Turn the raw dataset into small fixed-size arrays.
 - keeps only labeled wafers
 - resizes every wafer map to 64x64 (2 channels)
 - makes a stratified 70/15/15 train/val/test split
 - saves data/wafer_64.npz  (images, labels, split, lot name, wafer number)

    python preprocess.py                  # keeps at most 20000 "none" wafers (faster training)
    python preprocess.py --max_none 0     # keep ALL "none" wafers (172k total, slow on CPU)
"""
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from utils import load_lswmd, wafer_to_array, CLASS_TO_IDX, NONE_IDX, NPZ_PATH, set_seed

p = argparse.ArgumentParser()
p.add_argument("--max_none", type=int, default=20000,
               help="max number of 'none' (good) wafers to keep, 0 = keep all")
args = p.parse_args()
set_seed(42)

df = load_lswmd()
df = df.dropna(subset=["label"])
df = df[df["label"].isin(CLASS_TO_IDX)].reset_index(drop=True)
y = df["label"].map(CLASS_TO_IDX).values.astype(np.int64)
print("Labeled wafers:", len(df))

keep = np.arange(len(df))
if args.max_none > 0:
    rng = np.random.default_rng(42)
    none_idx = np.where(y == NONE_IDX)[0]
    if len(none_idx) > args.max_none:
        drop = rng.choice(none_idx, len(none_idx) - args.max_none, replace=False)
        keep = np.setdiff1d(keep, drop)
    print(f"Kept {int((y[keep] == NONE_IDX).sum())} 'none' wafers (cap = {args.max_none})")

maps = df["waferMap"].values
X = np.stack([wafer_to_array(maps[i]) for i in keep])
y = y[keep]
lot = np.array(df["lot"].astype(str).tolist())[keep]   # lot name of every kept wafer (used by the agent)
widx = df["widx"].values[keep].astype(np.int16)  # wafer number inside its lot
print("X shape:", X.shape, X.dtype)

idx = np.arange(len(y))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)
split = np.zeros(len(y), dtype=np.int8)      # 0 train, 1 val, 2 test
split[va] = 1
split[te] = 2
print("train / val / test:", len(tr), len(va), len(te))

np.savez_compressed(NPZ_PATH, X=X, y=y, split=split, lot=lot, widx=widx)
print("Saved", NPZ_PATH)
