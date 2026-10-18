"""
AGENT STEP 1 - The tools.

A "tool" is just a normal Python function. The LLM never sees the wafer images:
it only decides WHICH function to call, and reads the dictionary the function returns.
All numbers in the final answer therefore come from the CNN / the data, not from the LLM.

Tools (the LLM can call these):
    list_lots(min_wafers, limit)      find lots that have enough wafers
    query_lot_data(lot)               list the wafers of one lot (+ recorded labels)
    classify_wafer(wafer_id)          CNN prediction for one wafer
    summarize_lot(lot)                CNN on every wafer of a lot -> counts, PASS/FAIL, low-confidence
    explain_wafer(wafer_id)           Grad-CAM heatmap for one wafer
    flag_for_review(wafer_ids, why)   put wafers in outputs/review_queue.csv for a human

Try the tools without any LLM:
    python wafer_tools.py
"""
import os
import csv
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image

from model import WaferCNN
from gradcam import GradCAM, overlay
from utils import NPZ_PATH, MODEL_PATH, OUT_DIR, CLASSES, NONE_IDX, array_to_rgb, get_device

LOW_CONFIDENCE = 0.60                                   # below this -> a human should check
HEATMAP_DIR = os.path.join(OUT_DIR, "agent_heatmaps")
REVIEW_FILE = os.path.join(OUT_DIR, "review_queue.csv")

# ------------------------------------------------------------------ load data + model (once)
_d = np.load(NPZ_PATH)
if "lot" not in _d.files:
    raise SystemExit("data/wafer_64.npz has no lot information.\n"
                     "Run `python preprocess.py` again (it now saves lot names), then retry.")
X = _d["X"]                                             # (N, 2, 64, 64) uint8
table = pd.DataFrame({                                  # one row per wafer, like an Excel sheet
    "wafer_id": np.arange(len(X)),                      # = row number in X
    "lot": _d["lot"],
    "wafer_index": _d["widx"],
    "recorded_label": [CLASSES[i] for i in _d["y"]],    # label given in the WM-811K dataset
})

device = get_device()
_ckpt = torch.load(MODEL_PATH, map_location=device)
model = WaferCNN(num_classes=len(CLASSES)).to(device)
model.load_state_dict(_ckpt["model_state"])
model.eval()
cam_tool = GradCAM(model, model.features[-1])           # Grad-CAM on the last conv layer

_cache = {}                                             # wafer_id -> class probabilities


# ------------------------------------------------------------------ small helpers
def _norm_lot(lot):
    """'Lot 5', '5', 'LOT5' -> 'lot5'"""
    s = str(lot).strip().lower().replace(" ", "")
    return "lot" + s if s.isdigit() else s


def _lot_rows(lot):
    """Return the table rows of one lot, or None if the lot does not exist."""
    rows = table[table["lot"] == _norm_lot(lot)]
    return rows if len(rows) > 0 else None


def _predict(ids):
    """CNN probabilities for a list of wafer ids -> array (len(ids), 9). Uses a cache."""
    need = [i for i in ids if i not in _cache]
    for s in range(0, len(need), 256):
        chunk = need[s:s + 256]
        xb = torch.from_numpy(X[chunk]).float().div(255.0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(xb), dim=1).cpu().numpy()
        for i, p in zip(chunk, probs):
            _cache[i] = p
    return np.stack([_cache[i] for i in ids])


def _wafer_row(wafer_id):
    wafer_id = int(wafer_id)
    if wafer_id < 0 or wafer_id >= len(table):
        raise ValueError(f"wafer_id must be between 0 and {len(table) - 1}")
    return wafer_id, table.iloc[wafer_id]


def _describe(wafer_id, probs):
    """The CNN result for one wafer as a small dictionary."""
    pred = int(probs.argmax())
    row = table.iloc[wafer_id]
    return {
        "wafer_id": int(wafer_id),
        "lot": row["lot"],
        "wafer_index": int(row["wafer_index"]),
        "predicted_class": CLASSES[pred],
        "confidence": round(float(probs[pred]), 3),
        "status": "PASS" if pred == NONE_IDX else "FAIL",
        "low_confidence": bool(probs[pred] < LOW_CONFIDENCE),
        "recorded_label": row["recorded_label"],
    }


# ------------------------------------------------------------------ the 6 tools
def list_lots(min_wafers=5, limit=10):
    """Lots that have at least `min_wafers` wafers, biggest first."""
    counts = table.groupby("lot").size().reset_index(name="n_wafers")
    counts = counts[counts["n_wafers"] >= int(min_wafers)]
    counts = counts.sort_values(["n_wafers", "lot"], ascending=[False, True])
    return {"total_lots_found": int(len(counts)),
            "lots": [{"lot": r.lot, "n_wafers": int(r.n_wafers)}
                     for r in counts.head(int(limit)).itertuples()]}


def query_lot_data(lot):
    """List the wafers of one lot with their recorded (dataset) labels."""
    rows = _lot_rows(lot)
    if rows is None:
        return {"error": f"lot '{lot}' not found. Use list_lots to see valid lot names."}
    return {"lot": _norm_lot(lot), "n_wafers": int(len(rows)),
            "recorded_label_counts": rows["recorded_label"].value_counts().to_dict(),
            "wafers": [{"wafer_id": int(r.wafer_id), "wafer_index": int(r.wafer_index),
                        "recorded_label": r.recorded_label}
                       for r in rows.head(30).itertuples()]}


def classify_wafer(wafer_id):
    """Run the CNN on one wafer."""
    wafer_id, _ = _wafer_row(wafer_id)
    probs = _predict([wafer_id])[0]
    out = _describe(wafer_id, probs)
    top3 = probs.argsort()[::-1][:3]
    out["top3"] = {CLASSES[i]: round(float(probs[i]), 3) for i in top3}
    return out


def summarize_lot(lot):
    """Run the CNN on every wafer of a lot and summarise."""
    rows = _lot_rows(lot)
    if rows is None:
        return {"error": f"lot '{lot}' not found. Use list_lots to see valid lot names."}
    ids = [int(i) for i in rows["wafer_id"]]
    probs = _predict(ids)
    results = [_describe(i, p) for i, p in zip(ids, probs)]
    low = [r for r in results if r["low_confidence"]]
    class_counts = pd.Series([r["predicted_class"] for r in results]).value_counts().to_dict()
    n_agree = sum(r["predicted_class"] == r["recorded_label"] for r in results)
    return {"lot": _norm_lot(lot), "n_wafers": len(results),
            "n_pass": sum(r["status"] == "PASS" for r in results),
            "n_fail": sum(r["status"] == "FAIL" for r in results),
            "predicted_class_counts": class_counts,
            "mean_confidence": round(float(np.mean([r["confidence"] for r in results])), 3),
            "n_low_confidence": len(low),
            "low_confidence_wafers": [{"wafer_id": r["wafer_id"], "predicted_class": r["predicted_class"],
                                       "confidence": r["confidence"]} for r in low[:15]],
            "n_agree_with_recorded_label": n_agree}


def explain_wafer(wafer_id):
    """Grad-CAM: save a picture (wafer | heatmap) and say where the model looked."""
    wafer_id, _ = _wafer_row(wafer_id)
    arr = X[wafer_id]
    x = torch.from_numpy(arr).float().div(255.0).unsqueeze(0).to(device)
    cam, probs, pred = cam_tool.explain(x)               # cam: 64x64 numbers between 0 and 1

    # where is the heat? average distance from the wafer centre (0 = centre, 1 = edge)
    # of the HOTTEST 10% of pixels (the blurry blob around them would pull the average outwards)
    yy, xx = np.mgrid[0:cam.shape[0], 0:cam.shape[1]]
    r = np.sqrt((yy - 31.5) ** 2 + (xx - 31.5) ** 2) / 32.0
    hot = cam >= np.quantile(cam, 0.90)
    mean_r = float(r[hot].mean()) if cam.max() > 0 else None
    focus = "unclear" if mean_r is None else "center" if mean_r < 0.35 else "middle" if mean_r < 0.65 else "edge"

    rgb = array_to_rgb(arr)
    both = np.concatenate([rgb, overlay(rgb, cam)], axis=1)      # wafer | wafer + heatmap
    img = Image.fromarray(both).resize((both.shape[1] * 4, both.shape[0] * 4), Image.NEAREST)
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    path = os.path.join(HEATMAP_DIR, f"wafer_{wafer_id}.png")
    img.save(path)

    out = _describe(wafer_id, probs)
    out["heatmap_path"] = path
    out["model_focus"] = focus          # center / middle / edge
    out["note"] = "Grad-CAM shows where the model looked. It is not proof of the physical cause."
    return out


def flag_for_review(wafer_ids, reason="low confidence"):
    """Add wafers to the human review queue (outputs/review_queue.csv)."""
    if isinstance(wafer_ids, (int, str)):
        wafer_ids = [wafer_ids]
    ids = [_wafer_row(i)[0] for i in wafer_ids]
    probs = _predict(ids)
    os.makedirs(os.path.dirname(REVIEW_FILE), exist_ok=True)
    new_file = not os.path.exists(REVIEW_FILE)
    with open(REVIEW_FILE, "a", newline="") as f:
        wr = csv.writer(f)
        if new_file:
            wr.writerow(["time", "wafer_id", "lot", "wafer_index", "predicted_class", "confidence", "reason"])
        for i, p in zip(ids, probs):
            d = _describe(i, p)
            wr.writerow([datetime.now().isoformat(timespec="seconds"), i, d["lot"], d["wafer_index"],
                         d["predicted_class"], d["confidence"], reason])
    return {"flagged": len(ids), "wafer_ids": ids, "review_file": REVIEW_FILE}


# ------------------------------------------------------------------ registry + descriptions for the LLM
TOOLS = {"list_lots": list_lots, "query_lot_data": query_lot_data, "classify_wafer": classify_wafer,
         "summarize_lot": summarize_lot, "explain_wafer": explain_wafer, "flag_for_review": flag_for_review}


def _tool(name, description, properties, required):
    """Build one tool description in the format LLM servers expect (JSON schema)."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


SCHEMAS = [
    _tool("list_lots", "Find lots that have at least min_wafers wafers (biggest first). "
          "Use this when the user does not name a lot.",
          {"min_wafers": {"type": "integer", "description": "minimum wafers per lot, default 5"},
           "limit": {"type": "integer", "description": "how many lots to return, default 10"}}, []),
    _tool("query_lot_data", "List the wafers of one lot with their recorded labels from the dataset.",
          {"lot": {"type": "string", "description": "lot name such as 'lot12'"}}, ["lot"]),
    _tool("classify_wafer", "Run the CNN on ONE wafer: predicted defect class, confidence, PASS/FAIL.",
          {"wafer_id": {"type": "integer", "description": "wafer id"}}, ["wafer_id"]),
    _tool("summarize_lot", "Run the CNN on ALL wafers of a lot: PASS/FAIL counts, predicted class counts, "
          "and which wafers have low confidence.",
          {"lot": {"type": "string", "description": "lot name such as 'lot12'"}}, ["lot"]),
    _tool("explain_wafer", "Make a Grad-CAM heatmap for ONE wafer and say where the model looked "
          "(center / middle / edge).",
          {"wafer_id": {"type": "integer", "description": "wafer id"}}, ["wafer_id"]),
    _tool("flag_for_review", "Put wafers in the human review queue. Use it for low-confidence wafers.",
          {"wafer_ids": {"type": "array", "items": {"type": "integer"}, "description": "wafer ids to flag"},
           "reason": {"type": "string", "description": "short reason"}}, ["wafer_ids"]),
]


def run_tool(name, args):
    """Run one tool by name. Never crashes: errors go back to the LLM as {'error': ...}."""
    if name not in TOOLS:
        return {"error": f"unknown tool '{name}'. Available: {list(TOOLS)}"}
    try:
        return TOOLS[name](**args)
    except TypeError as e:
        return {"error": f"bad arguments: {e}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # quick self-test, no LLM needed
    print("wafers:", len(table), "| lots:", table["lot"].nunique())
    biggest = list_lots(min_wafers=5, limit=3)
    print("list_lots      ->", json.dumps(biggest))
    lot = biggest["lots"][0]["lot"]
    s = summarize_lot(lot)
    print("summarize_lot  ->", json.dumps(s)[:400])
    print("classify_wafer ->", json.dumps(classify_wafer(0)))
    print("explain_wafer  ->", json.dumps(explain_wafer(0)))
