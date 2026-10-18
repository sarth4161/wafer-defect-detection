"""
PHASE 2 + 3 - Train the CNN from scratch.
Saves the best model (by validation macro-F1) to outputs/best_model.pt

    python train.py
    python train.py --epochs 20 --batch 128 --lr 0.001
"""
import argparse
import csv
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import WaferCNN
from utils import NPZ_PATH, MODEL_PATH, OUT_DIR, CLASSES, IMG_SIZE, set_seed, get_device


class WaferDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float() / 255.0
        if self.augment:   # flips / 90-degree turns keep the defect type the same
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[2])
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[1])
            x = torch.rot90(x, k=int(np.random.randint(4)), dims=[1, 2])
        return x, int(self.y[i])


def run_epoch(model, loader, device, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds.append(out.argmax(1).cpu().numpy())
            labels.append(y.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return (total_loss / len(labels), float((preds == labels).mean()),
            f1_score(labels, preds, average="macro", zero_division=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    set_seed(42)
    device = get_device()
    print("Device:", device)
    os.makedirs(OUT_DIR, exist_ok=True)

    d = np.load(NPZ_PATH)
    X, y, split = d["X"], d["y"], d["split"]
    tr, va = split == 0, split == 1
    train_ds = WaferDataset(X[tr], y[tr], augment=True)
    val_ds = WaferDataset(X[va], y[va])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    # class weights: rare defect classes get a bigger weight (sqrt of inverse frequency)
    counts = np.bincount(y[tr], minlength=len(CLASSES)).astype(np.float64)
    w = (counts.sum() / (len(CLASSES) * np.maximum(counts, 1))) ** 0.5
    w = w / w.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    model = WaferCNN(num_classes=len(CLASSES)).to(device)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        tl, ta, tf = run_epoch(model, train_loader, device, loss_fn, optimizer)
        vl, va_acc, vf = run_epoch(model, val_loader, device, loss_fn)
        scheduler.step()
        history.append([epoch, tl, ta, tf, vl, va_acc, vf])
        mark = ""
        if vf > best_f1:
            best_f1 = vf
            torch.save({"model_state": model.state_dict(), "classes": CLASSES,
                        "img_size": IMG_SIZE}, MODEL_PATH)
            mark = "  <- saved"
        print(f"Epoch {epoch:2d}/{args.epochs} | train loss {tl:.3f} acc {ta:.3f} f1 {tf:.3f} "
              f"| val loss {vl:.3f} acc {va_acc:.3f} macro-F1 {vf:.3f}{mark}")

    with open(os.path.join(OUT_DIR, "training_log.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"])
        wr.writerows(history)

    h = np.array(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h[:, 0], h[:, 1], label="train"); ax[0].plot(h[:, 0], h[:, 4], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h[:, 0], h[:, 3], label="train"); ax[1].plot(h[:, 0], h[:, 6], label="val")
    ax[1].set_title("Macro-F1"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=120)
    print(f"\nBest validation macro-F1: {best_f1:.4f}  -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
