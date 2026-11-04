#!/usr/bin/env python
"""Evaluate the trained waste classifier on the held-out test split.

Reports accuracy, macro/weighted precision/recall/F1, per-class metrics, a
confusion matrix and single-image inference latency — computed ONLY from the
independent test split written by prepare_dataset.py. If weights, config or
the test split are missing, it prints that evaluation is pending and exits
non-zero instead of inventing numbers.

    python backend/evaluate.py --split backend/data/split.json --models-dir backend/models
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from train import SplitDataset
from waste_model import build_model, load_model, preprocess
from torch.utils.data import DataLoader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=Path(__file__).parent / "data" / "split.json")
    ap.add_argument("--models-dir", type=Path, default=Path(__file__).parent / "models")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    try:
        model, config = load_model(args.models_dir)
    except FileNotFoundError as e:
        print(f"EVALUATION PENDING: {e}")
        print("Train the model first: python backend/train.py --split backend/data/split.json")
        raise SystemExit(1)

    manifest = json.loads(args.split.read_text())
    test_rows = manifest["test"]
    if not test_rows:
        print("EVALUATION PENDING: the split manifest has no test rows — re-run prepare_dataset.py.")
        raise SystemExit(1)
    classes: list[str] = config["classes"]
    test_ds = SplitDataset(test_rows, classes, config, train=False)
    loader = DataLoader(test_ds, batch_size=args.batch, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(1).cpu()
            all_preds.extend(int(p) for p in preds)
            all_labels.extend(int(l) for l in labels)

    try:
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
    except ImportError:
        raise SystemExit("Missing dependency. Run: pip install scikit-learn")

    accuracy = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, support = precision_recall_fscore_support(all_labels, all_preds, labels=range(len(classes)), zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))

    print(f"\nHeld-out TEST metrics — {len(test_rows)} images, {len(classes)} classes "
          f"(dataset: {config.get('dataset_name')}, trained {config.get('trained_at')})")
    print(f"accuracy          : {accuracy:.4f}")
    print(f"precision (macro) : {p_macro:.4f}   recall (macro): {r_macro:.4f}   F1 (macro): {f1_macro:.4f}")
    print(f"precision (weight): {p_w:.4f}   recall (weight): {r_w:.4f}   F1 (weighted): {f1_w:.4f}")
    print("\nper-class (dataset classes):")
    print(f"{'class':<14}{'prec':>8}{'rec':>8}{'f1':>8}{'support':>10}")
    for i, c in enumerate(classes):
        print(f"{c:<14}{p_cls[i]:>8.4f}{r_cls[i]:>8.4f}{f1_cls[i]:>8.4f}{int(support[i]):>10}")
    print("\nconfusion matrix (rows = true, cols = predicted):")
    header = " " * 14 + "".join(f"{c[:8]:>10}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        print(f"{c:<14}" + "".join(f"{v:>10}" for v in cm[i]))

    # Single-image inference latency (preprocessing + forward pass, CPU/GPU as available)
    sample = Image.open(test_rows[0]["path"]).convert("RGB")
    tensor = preprocess(config)(sample).unsqueeze(0).to(device)
    torch.set_grad_enabled(False)
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        model(tensor)
        latencies.append((time.perf_counter() - start) * 1000)
    latencies.sort()
    print(f"\ninference latency over 100 runs on {device}: "
          f"mean={statistics.fmean(latencies):.1f}ms p50={latencies[49]:.1f}ms p95={latencies[94]:.1f}ms")

    report = {
        "dataset": config.get("dataset_name"),
        "test_images": len(test_rows),
        "accuracy": round(accuracy, 4),
        "macro": {"precision": round(p_macro, 4), "recall": round(r_macro, 4), "f1": round(f1_macro, 4)},
        "weighted": {"precision": round(p_w, 4), "recall": round(r_w, 4), "f1": round(f1_w, 4)},
        "per_class": {c: {"precision": round(p_cls[i], 4), "recall": round(r_cls[i], 4),
                          "f1": round(f1_cls[i], 4), "support": int(support[i])} for i, c in enumerate(classes)},
        "confusion_matrix": {"classes": classes, "rows_true_cols_pred": cm.tolist()},
        "latency_ms": {"mean": round(statistics.fmean(latencies), 1), "p50": round(latencies[49], 1), "p95": round(latencies[94], 1)},
    }
    out = args.models_dir / "evaluation_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {out} — copy these numbers into the README's Evaluation section.")


if __name__ == "__main__":
    main()
