"""Shared helpers: paths, class names, dataset loading, image <-> wafer conversion."""
import os
import sys
import random
import importlib

import numpy as np
from PIL import Image
from matplotlib import cm

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
SAMPLES_DIR = os.path.join(ROOT, "samples")
PKL_PATH = os.path.join(DATA_DIR, "LSWMD.pkl")        # original WM-811K file
NPZ_PATH = os.path.join(DATA_DIR, "wafer_64.npz")     # our preprocessed cache
MODEL_PATH = os.path.join(OUT_DIR, "best_model.pt")

# ---------------------------------------------------------------- classes
# WM-811K has 8 defect patterns + "none" (no defect)
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Random", "Scratch", "Near-full", "none"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NONE_IDX = CLASS_TO_IDX["none"]       # "none" = good wafer -> PASS
IMG_SIZE = 64


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- loading WM-811K
def get_label(x):
    """failureType cells look like array([['Center']]) or an empty array (= unlabeled)."""
    try:
        arr = np.asarray(x, dtype=object).flatten()
        if arr.size == 0:
            return None
        return str(arr[0])
    except Exception:
        return None


def _add_old_pandas_aliases():
    """LSWMD.pkl was saved with an old pandas that used 'pandas.indexes'."""
    import pandas.core.indexes as idx
    sys.modules.setdefault("pandas.indexes", idx)
    for sub in ["base", "numeric", "range", "multi", "category", "datetimes"]:
        try:
            mod = importlib.import_module("pandas.core.indexes." + sub)
            sys.modules.setdefault("pandas.indexes." + sub, mod)
        except ImportError:
            pass


def load_lswmd(path=PKL_PATH):
    """Load the WM-811K pickle. Returns DataFrame with columns waferMap, label (str or None)."""
    import pandas as pd
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download LSWMD.pkl (WM-811K) from Kaggle and put it in the data/ folder.\n"
            "See README.md, Step 2.")
    try:
        df = pd.read_pickle(path)
    except (ModuleNotFoundError, AttributeError):
        _add_old_pandas_aliases()
        try:
            df = pd.read_pickle(path)
        except Exception as e:
            raise RuntimeError(
                "Could not read LSWMD.pkl with your pandas version.\n"
                "Fix: use Python 3.9-3.11 with `pip install \"pandas<2\"` (or run on Google Colab/Kaggle),\n"
                f"original error: {e}")
    df = df[["waferMap", "failureType"] + [c for c in ("lotName", "waferIndex") if c in df.columns]].copy()
    df["label"] = df["failureType"].apply(get_label)
    # lot name (like "lot12") and wafer number inside the lot (1..25) - used by the agent
    df["lot"] = df["lotName"].apply(get_label) if "lotName" in df.columns else "unknown"
    df["widx"] = (pd.to_numeric(df["waferIndex"], errors="coerce").fillna(0).astype(int)
                  if "waferIndex" in df.columns else 0)
    return df[["waferMap", "label", "lot", "widx"]]


# ---------------------------------------------------------------- wafer map -> network input
def wafer_to_array(w, size=IMG_SIZE):
    """
    w: 2D array with values 0 (outside wafer), 1 (good die), 2 (defective die).
    Returns uint8 array (2, size, size):
      channel 0 = where dies exist, channel 1 = where dies are defective.
    Area-averaging (BOX) keeps thin scratches visible after resizing.
    """
    w = np.asarray(w)
    chans = []
    for mask in (w > 0, w == 2):
        im = Image.fromarray((mask.astype(np.uint8)) * 255)
        im = im.resize((size, size), Image.BOX)
        chans.append(np.asarray(im))
    return np.stack(chans).astype(np.uint8)


def array_to_rgb(arr):
    """(2,H,W) network input -> RGB uint8 picture (purple=outside, teal=good die, yellow=defect)."""
    a = arr.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    v = 0.5 * a[0] + 0.5 * a[1]
    return (cm.viridis(v)[..., :3] * 255).astype(np.uint8)


def wafer_to_rgb(w, out_size=256):
    """Original wafer map (0/1/2) -> clean RGB image (used to export sample PNGs)."""
    w = np.asarray(w)
    lut = np.array([[68, 1, 84], [33, 145, 140], [253, 231, 37]], dtype=np.uint8)
    img = Image.fromarray(lut[np.clip(w, 0, 2)])
    h, wd = w.shape
    scale = max(1, out_size // max(h, wd))
    return img.resize((wd * scale, h * scale), Image.NEAREST)


# ---------------------------------------------------------------- uploaded image -> wafer map
# Supported colour styles: matplotlib viridis (default) and jet, plus black/white and gray images.
_PALETTE = [
    # (R, G, B), class
    ((68, 1, 84), 0), ((0, 0, 0), 0), ((255, 255, 255), 0), ((0, 0, 128), 0),
    ((33, 145, 140), 1), ((127, 255, 127), 1),
    ((253, 231, 37), 2), ((128, 0, 0), 2),
]


def image_to_wafer(img):
    """PIL image -> 2D int array with 0 (outside), 1 (good die), 2 (defective die)."""
    rgb = np.asarray(img.convert("RGB")).astype(np.int32)
    is_gray = (np.abs(rgb[..., 0] - rgb[..., 1]).max() < 12 and
               np.abs(rgb[..., 1] - rgb[..., 2]).max() < 12)
    if is_gray:
        g = rgb[..., 0]
        return np.where(g < 64, 0, np.where(g < 192, 1, 2)).astype(np.uint8)
    colors = np.array([c for c, _ in _PALETTE], dtype=np.int32)       # (K,3)
    classes = np.array([k for _, k in _PALETTE], dtype=np.uint8)      # (K,)
    d = ((rgb[:, :, None, :] - colors[None, None, :, :]) ** 2).sum(-1)  # (H,W,K)
    return classes[d.argmin(-1)]
