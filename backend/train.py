#!/usr/bin/env python
"""Train the EcoLens waste classifier (MobileNetV3-Small transfer learning).

Reads the fixed split manifest produced by prepare_dataset.py (no leakage:
the split is image-level and reused for evaluation), fine-tunes an
ImageNet-pretrained MobileNetV3-Small, keeps the best validation checkpoint
plus a preprocessing/config JSON, and writes a training summary.

    python backend/train.py --split backend/data/split.json --epochs 25 --batch 32

Only the validation set guides early stopping. Test metrics come exclusively
from evaluate.py on the held-out test split.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from waste_model import IMAGENET_MEAN, IMAGENET_STD, build_model, preprocess


class SplitDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], classes: list[str], config: dict, train: bool):
        self.rows = rows
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = preprocess(config, train=train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), self.class_to_idx[row["dataset_class"]]


def top1_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.numel()
    return correct / max(1, total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=Path(__file__).parent / "data" / "split.json")
    ap.add_argument("--models-dir", type=Path, default=Path(__file__).parent / "models")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience on val accuracy")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    manifest = json.loads(args.split.read_text())
    classes: list[str] = manifest["classes"]
    config = {
        "arch": "mobilenet_v3_small",
        "num_classes": len(classes),
        "classes": classes,
        "idx_to_class": {i: c for i, c in enumerate(classes)},
        "class_to_category": manifest["class_to_category"],
        "image_size": 224,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "uncertain_threshold": 55,
        "dataset_name": Path(manifest["dataset_dir"]).name,
        "dataset_licences": manifest["licences"],
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | classes: {classes}")

    train_ds = SplitDataset(manifest["train"], classes, config, train=True)
    val_ds = SplitDataset(manifest["val"], classes, config, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")

    model = build_model(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_val, best_epoch, bad_epochs = 0.0, -1, 0
    history = []
    args.models_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.numel()
            seen += labels.numel()
        scheduler.step()
        val_acc = top1_accuracy(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": round(running_loss / max(1, seen), 4), "val_acc": round(val_acc, 4)})
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={history[-1]['train_loss']}  val_acc={history[-1]['val_acc']}")

        if val_acc > best_val:
            best_val, best_epoch, bad_epochs = val_acc, epoch, 0
            torch.save(model.state_dict(), args.models_dir / "best.pt")
            (args.models_dir / "config.json").write_text(json.dumps(config, indent=2))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch + 1} (no val improvement for {args.patience} epochs).")
                break

    if best_epoch < 0:
        raise SystemExit("Validation accuracy never improved — nothing saved. Check the dataset manifest.")

    summary = {
        "best_val_acc": round(best_val, 4),
        "best_epoch": best_epoch + 1,
        "epochs_run": len(history),
        "history": history,
        "test_metrics": "PENDING — run evaluate.py; test metrics are produced only from the held-out test split.",
    }
    (args.models_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Best val accuracy {best_val:.4f} (epoch {best_epoch + 1}). Weights + config saved to {args.models_dir}.")
    print("Run evaluate.py for real held-out test metrics before quoting any accuracy.")


if __name__ == "__main__":
    main()
