"""
PHASE 3 - Evaluate the trained CNN on the test set.
Prints precision / recall / F1 per class, macro-F1, and PASS/FAIL results.
Saves outputs/confusion_matrix.png and outputs/test_results.json

    python evaluate.py
"""
import json
import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import WaferCNN
from utils import NPZ_PATH, MODEL_PATH, OUT_DIR, CLASSES, NONE_IDX, get_device

device = get_device()
ckpt = torch.load(MODEL_PATH, map_location=device)
model = WaferCNN(num_classes=len(CLASSES)).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

d = np.load(NPZ_PATH)
te = d["split"] == 2
X, y = d["X"][te], d["y"][te]

preds = []
with torch.no_grad():
    for i in range(0, len(X), 256):
        xb = torch.from_numpy(X[i:i + 256]).float().div(255.0).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
preds = np.concatenate(preds)

print(f"Test samples: {len(y)}")
print(f"Accuracy : {accuracy_score(y, preds):.4f}")
print(f"Macro-F1 : {f1_score(y, preds, average='macro', zero_division=0):.4f}\n")
print("Per-class results (9 classes):")
print(classification_report(y, preds, labels=range(len(CLASSES)), target_names=CLASSES,
                            digits=3, zero_division=0))

# PASS / FAIL view: "none" = PASS, any defect class = FAIL
y_bin, p_bin = (y != NONE_IDX).astype(int), (preds != NONE_IDX).astype(int)
print("PASS / FAIL results (FAIL = any defect pattern):")
print(classification_report(y_bin, p_bin, target_names=["PASS", "FAIL"], digits=3, zero_division=0))

# confusion matrix (rows normalised)
cm = confusion_matrix(y, preds, labels=range(len(CLASSES)))
cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=120)

rep = classification_report(y, preds, labels=range(len(CLASSES)), target_names=CLASSES,
                            output_dict=True, zero_division=0)
rep_bin = classification_report(y_bin, p_bin, target_names=["PASS", "FAIL"], output_dict=True, zero_division=0)
with open(os.path.join(OUT_DIR, "test_results.json"), "w") as f:
    json.dump({"accuracy": float(accuracy_score(y, preds)), "per_class": rep, "pass_fail": rep_bin}, f, indent=2)
print("Saved outputs/confusion_matrix.png and outputs/test_results.json")
