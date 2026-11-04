#!/usr/bin/env python
"""Dataset preparation for the EcoLens AI classifier.

Downloads a labelled public waste dataset, maps its classes onto the app's six
categories, and writes a stratified image-level train/val/test split manifest.
The split is fixed once and reused by train.py and evaluate.py so no image can
appear in two splits (prevents data leakage).

Datasets
--------
1. trashnet (default) — Gary Thung & Mindy Yang, https://github.com/garythung/trashnet
   MIT License (verified in the repo's LICENSE file, Copyright (c) 2017 Gary Thung).
   2,527 images, 6 classes: cardboard, glass, metal, paper, plastic, trash.

2. garbage12 + GIZ e-waste (used for the shipped model) — the 12-class set
   below has no e-waste folder, so an "e-waste" folder is added from
   GIZ/E-Waste-Database on Hugging Face (CC BY 4.0, every image is e-waste).
   Run with --source dir on the merged folder.

   garbage12 — Kaggle "Garbage Classification (12 classes)" by mostafaabla,
   https://www.kaggle.com/datasets/mostafaabla/garbage-classification
   15,150 images, 12 classes. Verify the licence on the Kaggle page before any
   redistribution; use `--kaggle` after `pip install kaggle` and accepting the
   dataset's terms in your Kaggle account.

Limitations (documented, not hidden)
------------------------------------
- trashnet contains NO organic, hazardous or e-waste images: a model trained
  only on it can genuinely predict recyclable and residual items only.
- garbage12 covers biological (organic), battery (hazardous) and e-waste, but
  "hazardous" is represented by batteries alone and "uncertain" is never a
  dataset class — it is produced only by the confidence threshold.
- Photos in these datasets are studio-like, not real bin photos: expect an
  accuracy drop on real-world images. Report this in the demo.

Usage
-----
    python backend/prepare_dataset.py --source hf --data-dir backend/data
    python backend/prepare_dataset.py --source dir --dir /path/to/class_folders
    python backend/prepare_dataset.py --kaggle --data-dir backend/data
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from common import CATEGORIES

# dataset class name -> app category
TRASHNET_MAPPING = {
    "cardboard": "recyclable",
    "glass": "recyclable",
    "metal": "recyclable",
    "paper": "recyclable",
    "plastic": "recyclable",
    "trash": "residual",
}
GARBAGE12_MAPPING = {
    "battery": "hazardous",
    "biological": "organic",
    "brown-glass": "recyclable",
    "green-glass": "recyclable",
    "white-glass": "recyclable",
    "cardboard": "recyclable",
    "clothes": "residual",
    "e-waste": "ewaste",
    "metal": "recyclable",
    "paper": "recyclable",
    "plastic": "recyclable",
    "shoes": "residual",
    "trash": "residual",
}

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # test gets the remainder
SEED = 42
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def export_from_hf(out_dir: Path) -> Path:
    """Downloads garythung/trashnet from Hugging Face into class folders."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Missing dependency. Run: pip install datasets pillow") from e
    ds = load_dataset("garythung/trashnet", split="train")
    root = out_dir / "trashnet"
    counts: Counter[str] = Counter()
    for i, row in enumerate(ds):
        label = row["label"]
        class_name = ds.features["label"].int2str(label)
        target = root / class_name / f"{i:05d}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        row["image"].convert("RGB").save(target, format="JPEG", quality=90)
        counts[class_name] += 1
    print(f"Exported {sum(counts.values())} images from Hugging Face: {dict(counts)}")
    return root


def class_folders(dir: Path) -> tuple[Path, dict[str, str]]:
    """Reads a directory tree of class folders; asks which mapping applies."""
    classes = sorted(p.name for p in dir.iterdir() if p.is_dir() and any(p.iterdir()))
    if set(classes) <= set(TRASHNET_MAPPING):
        return dir, TRASHNET_MAPPING
    if set(classes) <= set(GARBAGE12_MAPPING):
        return dir, GARBAGE12_MAPPING
    raise SystemExit(
        f"Classes {classes} match neither TRASHNET_MAPPING nor GARBAGE12_MAPPING. "
        "Edit the MAPPING tables in prepare_dataset.py for your dataset."
    )


def download_from_kaggle(out_dir: Path) -> Path:
    """Downloads the 12-class dataset via the Kaggle CLI (user must accept terms)."""
    import subprocess
    root = out_dir / "garbage12_raw"
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "mostafaabla/garbage-classification", "-p", str(root), "--unzip"],
        check=True,
    )
    # the archive unpacks to garbage classification/Garbage classification/<class>/
    nested = [p for p in root.rglob("*") if p.is_dir() and any(c.is_dir() for c in p.iterdir())]
    if not nested:
        raise SystemExit("Unexpected Kaggle archive layout; inspect " + str(root))
    return nested[0]


def build_split(images_dir: Path, mapping: dict[str, str], out_dir: Path, max_per_class: int = 0) -> Path:
    rng = random.Random(SEED)
    by_class: defaultdict[str, list[Path]] = defaultdict(list)
    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in mapping:
            continue
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
        rng.shuffle(files)
        if max_per_class > 0:
            files = files[:max_per_class]  # seeded cap to limit class imbalance (e.g. 5k clothes)
        by_class[class_dir.name] = files

    if not by_class:
        raise SystemExit(f"No usable class folders with images under {images_dir}")

    split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for class_name, files in sorted(by_class.items()):
        n = len(files)
        n_train = round(n * TRAIN_FRAC)
        n_val = round(n * VAL_FRAC)
        bounds = {"train": files[:n_train], "val": files[n_train:n_train + n_val], "test": files[n_train + n_val:]}
        for subset, subset_files in bounds.items():
            for p in subset_files:
                split[subset].append({"path": str(p), "dataset_class": class_name, "category": mapping[class_name]})

    manifest = {
        "dataset_dir": str(images_dir),
        "class_to_category": mapping,
        "classes": sorted(mapping),
        "seed": SEED,
        "max_per_class": max_per_class,
        "counts": {s: {c: sum(1 for r in rows if r["dataset_class"] == c) for c in sorted(mapping)} for s, rows in split.items()},
        "sizes": {s: len(rows) for s, rows in split.items()},
        **split,  # the actual per-image rows used by train.py / evaluate.py
        "licences": {
            "trashnet": "MIT (c) 2017 Gary Thung — github.com/garythung/trashnet/LICENSE",
            "garbage12": "Kaggle mostafaabla/garbage-classification (12 classes, NO e-waste class) — check licence on Kaggle before redistribution",
            "giz_ewaste": "CC BY 4.0 — huggingface.co/datasets/GIZ/E-Waste-Database (all images are e-waste)",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "split.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"Split written to {path}: train={manifest['sizes']['train']} "
          f"val={manifest['sizes']['val']} test={manifest['sizes']['test']} "
          f"classes={sorted(mapping)}")
    for subset, counts in manifest["counts"].items():
        print(f"  {subset}: {counts}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "dir", "kaggle"], default="hf",
                    help="hf: download trashnet from Hugging Face; dir: use local class folders; "
                         "kaggle: download the 12-class dataset with the Kaggle CLI")
    ap.add_argument("--dir", type=Path, help="class-folder root when --source dir")
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--max-per-class", type=int, default=0, help="seeded cap per dataset class (0 = no cap)")
    args = ap.parse_args()

    if args.source == "hf":
        images_dir = export_from_hf(args.data_dir)
        mapping = TRASHNET_MAPPING
    elif args.source == "kaggle":
        images_dir = download_from_kaggle(args.data_dir)
        mapping = GARBAGE12_MAPPING
    else:
        if not args.dir:
            raise SystemExit("--source dir requires --dir /path/to/class_folders")
        images_dir, mapping = class_folders(args.dir)

    missing = set(CATEGORIES) - {"uncertain"} - set(mapping.values())
    if missing:
        print(f"WARNING: this dataset cannot train {sorted(missing)} — those categories "
              f"will never be predicted locally. See the Limitations section in backend/README.md.")
    build_split(images_dir, mapping, args.out_dir, args.max_per_class)


if __name__ == "__main__":
    main()
