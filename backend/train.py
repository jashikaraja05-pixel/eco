#!/usr/bin/env python
"""Train the EcoLens waste classifier (MobileNetV3-Small transfer learning).

Reads the fixed split manifest produced by prepare_dataset.py (no leakage: the
split is image-level and reused for evaluation), validates the manifest/training
contract, fine-tunes an ImageNet-pretrained MobileNetV3-Small, keeps the best
validation checkpoint plus a preprocessing/config JSON, and writes a training
summary that records the seed and every hyperparameter.

    python backend/train.py --split backend/data/split.json --epochs 25 --batch 32 --seed 42

Contract enforced before a single forward pass (see manifest.py):
  * num_classes == len(classes) == number of checkpoint output neurons;
  * classes come only from dataset folders that actually had images;
  * every output class has training images (no untrained neuron);
  * idx_to_class / class_to_category agree with classes;
  * no image appears in two splits;
  * every category the app serves is covered unless --allow-partial is given.

Only the validation set guides early stopping. Test metrics come exclusively
from evaluate.py on the held-out test split.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from manifest import (
    DEFAULT_SEED,
    ManifestError,
    build_model_config,
    validate_manifest,
)
from seeding import loader_generator, seed_everything
from waste_model import IMAGENET_MEAN, IMAGENET_STD, build_model, preprocess


class SplitDataset(Dataset):
    """Rows from one split of the manifest; labels come from the manifest classes."""

    def __init__(self, rows: list[dict[str, str]], classes: list[str], config: dict, train: bool):
        self.rows = rows
        self.classes = classes
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


def load_manifest(path: Path, allow_partial: bool, mapping_tables: tuple[dict[str, str], ...]) -> tuple[dict, dict]:
    """Reads and validates the split manifest. Returns (manifest, contract report)."""
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as e:
        raise SystemExit(f"No split manifest at {path}. Run prepare_dataset.py first.") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"Split manifest {path} is not valid JSON: {e}") from e

    declared_partial = bool(manifest.get("coverage", {}).get("partial_training")) or bool(manifest.get("partial_training"))
    try:
        report = validate_manifest(
            manifest,
            mapping_tables=mapping_tables,
            allow_partial=allow_partial or declared_partial,
        )
    except ManifestError as e:
        hint = ""
        if not (allow_partial or declared_partial):
            hint = ("\nThe manifest already declares partial_training=true (produced with "
                    "prepare_dataset.py --allow-partial); re-run training with --allow-partial to accept it.")
        raise SystemExit(f"Invalid split manifest {path}:\n{e}{hint}") from e
    return manifest, report


def training_config(manifest: dict, report: dict, args: argparse.Namespace, seed_report: dict) -> dict[str, Any]:
    """Model config written next to best.pt — derived from the validated report."""
    config = build_model_config(
        manifest,
        mapping_tables=(manifest.get("class_to_category") or {},),
        allow_partial=True,  # coverage already validated above
        image_size=args.image_size,
        uncertain_threshold=55,
        seed=args.seed,
        dataset_name=manifest.get("dataset_name") or Path(str(manifest.get("dataset_dir", "dataset"))).name,
        normalization={"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    )
    config.update({
        "arch": "mobilenet_v3_small",
        "pretrained": not args.no_pretrained,
        "num_classes": report["num_classes"],
        "classes": report["classes"],
        "idx_to_class": {str(i): c for i, c in report["idx_to_class"].items()},
        "class_to_category": report["class_to_category"],
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "patience": args.patience,
            "image_size": args.image_size,
        },
        "seeding": seed_report,
    })
    return config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=Path(__file__).parent / "data" / "split.json")
    ap.add_argument("--models-dir", type=Path, default=Path(__file__).parent / "models")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience on val accuracy")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="random seed for splits/init/shuffling (recorded in config.json and training_summary.json)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="accept a dataset that cannot serve every app category "
                         "(the manifest records the reduced coverage; the app reports it)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="train from scratch without ImageNet weights (no download)")
    args = ap.parse_args()

    manifest_path = args.split
    # The mapping used to build the manifest decides which dataset class is
    # required for a missing category, so read it from the manifest itself.
    raw = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    mapping_tables: tuple[dict[str, str], ...] = (raw.get("class_to_category") or {},)

    manifest, report = load_manifest(manifest_path, args.allow_partial, mapping_tables)
    classes: list[str] = report["classes"]

    if int(manifest.get("num_classes", len(classes))) != len(classes):
        raise SystemExit(
            f"Manifest num_classes={manifest.get('num_classes')} but it lists {len(classes)} classes {classes}. "
            "Re-run prepare_dataset.py so the output head matches the dataset."
        )

    seed = args.seed
    if int(manifest.get("seed", seed)) != seed:
        print(f"NOTE: manifest was split with seed {manifest.get('seed')}; training/init use --seed {seed}. "
              "Pass the same --seed to reproduce the exact split.")
    seed_report = seed_everything(seed)

    config = training_config(manifest, report, args, seed_report)

    if report["missing_categories"]:
        print("WARNING: partial training accepted. Categories that can never be predicted locally: "
              f"{sorted(report['missing_categories'])}")
        for category, needed in report["missing_categories"].items():
            print(f"  - '{category}' needs dataset class folder(s): {needed or 'any class mapped to it'}")
    print(f"Manifest OK: {report['num_classes']} classes {classes}; "
          f"split sizes {report['sizes']}; present categories {report['present_categories']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | seed: {seed}")

    train_ds = SplitDataset(manifest["train"], classes, config, train=True)
    val_ds = SplitDataset(manifest["val"], classes, config, train=False)
    if not manifest["val"]:
        raise SystemExit("Manifest has no validation rows; early stopping and checkpoint selection need val.")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", generator=loader_generator(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")

    model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)
    out_features = model.classifier[3].out_features
    if out_features != len(classes):
        raise SystemExit(f"Model head has {out_features} outputs but the manifest has {len(classes)} classes.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_val, best_epoch, bad_epochs = -1.0, -1, 0
    history: list[dict[str, Any]] = []
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
        history.append({"epoch": epoch + 1, "train_loss": round(running_loss / max(1, seen), 4),
                        "val_acc": round(val_acc, 4), "lr": round(optimizer.param_groups[0]["lr"], 8)})
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
        "seed": seed,
        "seeding": seed_report,
        "hyperparameters": config["hyperparameters"],
        "num_classes": config["num_classes"],
        "classes": config["classes"],
        "class_to_category": config["class_to_category"],
        "split_sizes": report["sizes"],
        "present_categories": report["present_categories"],
        "missing_categories": sorted(report["missing_categories"]),
        "partial_training": report["partial_training"],
        "dataset_name": config.get("dataset_name"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device), "deterministic": seed_report["deterministic_algorithms"]},
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
