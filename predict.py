"""
PHASE 5 - Inference: wafer image -> defect class, confidence, PASS/FAIL, Grad-CAM heatmap.

Command line:
    python predict.py samples/Center_0.png
"""
import sys
import numpy as np
import torch
from PIL import Image

from model import WaferCNN
from gradcam import GradCAM, overlay
from utils import (MODEL_PATH, CLASSES, NONE_IDX, IMG_SIZE,
                   image_to_wafer, wafer_to_array, array_to_rgb, get_device)

LOW_CONFIDENCE = 0.60


def load_model(path=MODEL_PATH):
    device = get_device()
    ckpt = torch.load(path, map_location=device)
    model = WaferCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device


def predict_image(model, device, pil_img):
    """Returns a dict with everything the app needs."""
    wafer = image_to_wafer(pil_img)                  # 0/1/2 map
    arr = wafer_to_array(wafer, IMG_SIZE)            # (2,64,64) uint8
    x = torch.from_numpy(arr).float().div(255.0).unsqueeze(0).to(device)

    cam, probs, pred = GradCAM(model, model.features[-1]).explain(x)
    rgb = array_to_rgb(arr)
    status = "PASS" if pred == NONE_IDX else "FAIL"
    return {
        "pred_class": CLASSES[pred],
        "confidence": float(probs[pred]),
        "status": status,
        "p_fail": float(1.0 - probs[NONE_IDX]),     # probability that the wafer has any defect
        "probs": {c: float(p) for c, p in zip(CLASSES, probs)},
        "wafer_rgb": rgb,
        "overlay": overlay(rgb, cam),
        "low_confidence": float(probs[pred]) < LOW_CONFIDENCE,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python predict.py <wafer_image.png>")
    model, device = load_model()
    res = predict_image(model, device, Image.open(sys.argv[1]))
    print(f"Result     : {res['status']}")
    print(f"Defect type: {res['pred_class']}  (confidence {res['confidence']:.1%})")
    print(f"P(defect)  : {res['p_fail']:.1%}")
    if res["low_confidence"]:
        print("Note       : low confidence - please check this wafer manually")
    Image.fromarray(res["overlay"]).resize((256, 256), Image.NEAREST).save("outputs/last_prediction_gradcam.png")
    print("Heatmap saved to outputs/last_prediction_gradcam.png")
