"""
PHASE 4 - Grad-CAM, written by hand.

Idea: take the last conv feature maps A_k, see how much the predicted class score
changes when each map changes (gradient, averaged = weight a_k), then
    heatmap = ReLU( sum_k a_k * A_k )
Bright areas = wafer regions that pushed the model towards its prediction.

Run this file to make outputs/gradcam_examples.png (2 test wafers per class):
    python gradcam.py
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def explain(self, x, class_idx=None):
        """x: (1,C,H,W) tensor. Returns heatmap (H,W) in 0..1, class probabilities, class used."""
        self.model.eval()
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            if class_idx is None:
                class_idx = int(logits.argmax(1))
            self.model.zero_grad()
            logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # (1,K,1,1)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, probs, class_idx


def overlay(rgb, cam, alpha=0.5):
    """rgb: (H,W,3) uint8 wafer picture, cam: (H,W) 0..1  ->  (H,W,3) uint8 overlay."""
    heat = cm.jet(cam)[..., :3]
    out = (1 - alpha) * (rgb / 255.0) + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from model import WaferCNN
    from utils import NPZ_PATH, MODEL_PATH, OUT_DIR, CLASSES, array_to_rgb, get_device

    device = get_device()
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = WaferCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])
    cam_tool = GradCAM(model, model.features[-1])

    d = np.load(NPZ_PATH)
    te = d["split"] == 2
    X, y = d["X"][te], d["y"][te]

    fig, axes = plt.subplots(len(CLASSES), 4, figsize=(9, 2.2 * len(CLASSES)))
    for r, c in enumerate(CLASSES):
        ids = np.where(y == r)[0][:2]
        for k in range(2):
            ax1, ax2 = axes[r, 2 * k], axes[r, 2 * k + 1]
            ax1.axis("off"); ax2.axis("off")
            if k < len(ids):
                arr = X[ids[k]]
                x = torch.from_numpy(arr).float().div(255.0).unsqueeze(0).to(device)
                cam, probs, pred = cam_tool.explain(x)
                rgb = array_to_rgb(arr)
                ax1.imshow(rgb)
                ax1.set_title(f"true: {c}", fontsize=8)
                ax2.imshow(overlay(rgb, cam))
                ax2.set_title(f"pred: {CLASSES[pred]} ({probs[pred]:.0%})", fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gradcam_examples.png")
    plt.savefig(path, dpi=110)
    print("Saved", path)
