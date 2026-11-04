#!/usr/bin/env python
"""Dataset preparation for the EcoLens AI classifier.

Downloads a labelled public waste dataset, maps its classes onto the app's waste
categories, and writes a stratified image-level train/val/test split manifest.
The split is fixed once and reused by train.py and evaluate.py so no image can
appear in two splits (prevents data leakage).

Classes are derived from the class folders that ACTUALLY contain images. A
mapping key whose folder is missing or empty is never turned into an output
class (a 13-key mapping over a 12-folder dataset used to create one permanently
untrained output neuron). Category coverage is validated before anything is
written: if the dataset cannot serve one of the app's five categories,
preparation stops with an error naming the category and the dataset class it
needs, unless --allow-partial explicitly opts into reduced coverage.

Datasets
--------
1. trashnet (default) — Gary Thung & Mindy Yang, https://github.com/garythung/trashnet
   MIT License (verified in the repo's LICENSE file, Copyright (c) 2017 Gary Thung).
   2,527 images, 6 classes: cardboard, glass, metal, paper, plastic, trash.

2. garbage12 + GIZ e-waste (used for the shipped model) — the 12-class Kaggle
   set below has no e-waste folder, so an "e-waste" folder must be added from
   GIZ/E-Waste-Database on Hugging Face (CC BY 4.0, every image is e-waste).
   Run with --source dir on the merged folder.

   A directory may also mix TrashNet and garbage12 folder names: MERGED_MAPPING
   is the union of both tables, so a merged tree that still contains TrashNet's
   single "glass" folder (next to brown-/green-/white-glass) is recognised and
   "glass" maps to the same "recyclable" category. Folder names are matched
   through a declared normalization (case and separator insensitive, e.g.
   "E-Waste" -> "e-waste"), never by guessing: a folder whose normalized name is
   not a mapping key is left out of the output classes and reported.

   garbage12 — Kaggle "Garbage Classification (12 classes)" by mostafaabla,
   https://www.kaggle.com/datasets/mostafaabla/garbage-classification
   15,150 images, 12 classes. Verify the licence on the Kaggle page before any
   redistribution; use `--kaggle` after `pip install kaggle` and accepting the
   dataset's terms in your Kaggle account.

Limitations (documented, not hidden)
------------------------------------
- trashnet contains NO organic, hazardous or e-waste images: a model trained
  only on it can genuinely predict recyclable and residual items only, so it
  requires --allow-partial (the manifest then records partial_training=true).
- garbage12 covers biological (organic), battery (hazardous) and e-waste, but
  "hazardous" is represented by batteries alone and "uncertain" is never a
  dataset class — it is produced only by the confidence threshold.
- Photos in these datasets are studio-like, not real bin photos: expect an
  accuracy drop on real-world images. Report this in the demo.

Usage
-----
    python backend/prepare_dataset.py --source hf --data-dir backend/data --allow-partial
    python backend/prepare_dataset.py --source dir --dir /path/to/class_folders
    python backend/prepare_dataset.py --kaggle --data-dir backend/data
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common import CATEGORIES
from manifest import (
    APP_CATEGORIES,
    DEFAULT_SEED,
    IMAGE_EXT,
    MIN_IMAGES_PER_CLASS,
    TRAIN_FRAC,
    VAL_FRAC,
    DatasetCoverageError,
    ManifestError,
    check_category_coverage,
    split_counts,
    validate_manifest,
)

# dataset class name -> app category
TRASHNET_MAPPING = {
    "cardboard": "recyclable",
    "glass": "recyclable",
    "metal": "recyclable",
    "paper": "recyclable",
    "plastic": "recyclable",
    "trash": "residual",
}
# NOTE: "e-waste" is a real requirement, not a folder that always exists in the
# 12-class Kaggle archive. If the folder is absent it must NOT become a class:
# build_split() only keeps folders that contain images, and coverage validation
# then fails (or requires --allow-partial) instead of silently training a
# dead output neuron.
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

# Union of both tables, used when one directory mixes TrashNet and garbage12
# folder names (e.g. the merged dataset that adds "e-waste" to the 12 Kaggle
# classes while still containing TrashNet's single "glass" folder). Every key
# maps to the same category it has in its own table: "glass", "brown-glass",
# "green-glass" and "white-glass" all stay "recyclable". Values are never
# changed by merging, and no new category is introduced.
MERGED_MAPPING = {**TRASHNET_MAPPING, **GARBAGE12_MAPPING}

# Mapping tables that class_folders() may pick, in preference order (the most
# specific table wins, so a pure TrashNet tree still reports TrashNet hints).
MAPPING_TABLES = {
    "trashnet": TRASHNET_MAPPING,
    "garbage12": GARBAGE12_MAPPING,
    "merged": MERGED_MAPPING,
}

# Explicit synonyms that case/separator normalization cannot reach: canonical
# mapping key -> accepted alternative folder spellings. An alias only renames a
# folder to a class that already exists in the active table; it never invents a
# class or a category. Anything else is reported as unrecognised.
CLASS_ALIASES = {
    "e-waste": ("electronic-waste", "electronicwaste", "e-trash"),
    "battery": ("batteries",),
}

SEED = DEFAULT_SEED


def normalize_class_name(name: str) -> str:
    """Case- and separator-insensitive form of a class/folder name.

    Only lowercasing and stripping of non-alphanumerics is applied, so
    "E-Waste", "e_waste", "e waste" and "e-waste" all normalize identically.
    Nothing else is guessed.
    """
    return "".join(ch for ch in name.strip().lower() if ch.isalnum())


def build_class_index(mapping: dict[str, str]) -> dict[str, str]:
    """normalized name -> canonical dataset class for one mapping table.

    Raises SystemExit if two different keys normalize to the same name (an
    ambiguous table would otherwise map a folder to an arbitrary class).
    """
    index: dict[str, str] = {}

    def add(name: str, canonical: str, source: str) -> None:
        key = normalize_class_name(name)
        if not key:
            raise SystemExit(f"Mapping/alias entry {name!r} ({source}) normalizes to an empty name.")
        if key in index and index[key] != canonical:
            raise SystemExit(
                f"Ambiguous class name {name!r} ({source}): it resolves to both "
                f"'{index[key]}' and '{canonical}'. Fix the mapping/alias tables."
            )
        index[key] = canonical

    for canonical in mapping:
        add(canonical, canonical, "mapping key")
    for canonical, aliases in CLASS_ALIASES.items():
        if canonical not in mapping:
            continue  # alias for a class this table does not contain
        for alias in aliases:
            add(alias, canonical, f"alias of '{canonical}'")
    return index


def resolve_class_name(folder_name: str, mapping: dict[str, str],
                       index: dict[str, str] | None = None) -> str | None:
    """Canonical dataset class for a real folder name, or None if undeclared.

    None means "do not classify this folder": callers must report it instead of
    mapping it to a category by guessing.
    """
    index = build_class_index(mapping) if index is None else index
    return index.get(normalize_class_name(folder_name))



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


def _image_files(class_dir: Path) -> list[Path]:
    """Usable image files in a class folder (non-empty, known extension)."""
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXT and p.stat().st_size > 0
    )


def resolve_folders(dir: Path, mapping: dict[str, str]) -> tuple[dict[str, Path], list[str], list[str]]:
    """Maps real folder names to canonical dataset classes for one mapping table.

    Returns ``(folder_by_class, unrecognized, ambiguous)``:
    * ``folder_by_class`` — canonical class -> the folder that provides it;
    * ``unrecognized`` — folders with images whose normalized name is not a key
      or alias of this table. They are NEVER assigned a category;
    * ``ambiguous`` — messages for two folders that resolve to the same class
      (e.g. ``glass`` and ``Glass``), which would silently merge two sources.
    """
    index = build_class_index(mapping)
    folder_by_class: dict[str, Path] = {}
    unrecognized: list[str] = []
    ambiguous: list[str] = []
    for folder in sorted(p for p in dir.iterdir() if p.is_dir() and _image_files(p)):
        canonical = index.get(normalize_class_name(folder.name))
        if canonical is None:
            unrecognized.append(folder.name)
            continue
        if canonical in folder_by_class:
            ambiguous.append(
                f"'{folder_by_class[canonical].name}' and '{folder.name}' both resolve to class '{canonical}'"
            )
            continue
        folder_by_class[canonical] = folder
    return folder_by_class, unrecognized, ambiguous


def class_folders(dir: Path) -> tuple[Path, dict[str, str]]:
    """Reads a directory tree of class folders; picks the mapping that applies.

    A folder name is accepted when it is a mapping key, a declared alias, or the
    same name with different case/separators. Nothing is classified by guesswork:
    if no table covers every folder, preparation stops and names the folders it
    could not resolve.
    """
    folders = sorted(p.name for p in dir.iterdir() if p.is_dir() and _image_files(p))
    if not folders:
        raise SystemExit(f"No class folders with usable images under {dir}")

    # Most specific table first: a pure TrashNet tree keeps TrashNet's coverage
    # hints, while a mixed tree falls through to MERGED_MAPPING.
    for table_name in ("trashnet", "garbage12", "merged"):
        mapping = MAPPING_TABLES[table_name]
        folder_by_class, unrecognized, ambiguous = resolve_folders(dir, mapping)
        if ambiguous:
            raise SystemExit(
                "Ambiguous dataset folders for mapping "
                f"'{table_name}': {'; '.join(ambiguous)}. Keep exactly one folder per class "
                "(rename or remove the duplicate) so no two sources are silently merged."
            )
        if not unrecognized:
            print(
                f"Using mapping '{table_name}': {len(folder_by_class)} class folder(s) recognised "
                f"-> {sorted(folder_by_class)}"
            )
            return dir, mapping

    folder_by_class, unrecognized, _ = resolve_folders(dir, MERGED_MAPPING)
    raise SystemExit(
        f"Unrecognised dataset folder(s) under {dir}: {unrecognized}. They are not mapped to any "
        "waste category and were not silently classified. Recognised class names are "
        f"{sorted(MERGED_MAPPING)} (plus declared aliases). Rename the folders to match a mapping "
        "key or add an alias to CLASS_ALIASES in prepare_dataset.py."
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


def build_split(images_dir: Path, mapping: dict[str, str], out_dir: Path, max_per_class: int = 0,
                allow_partial: bool = False, seed: int = SEED) -> Path:
    """Writes split.json built only from class folders that contain images.

    Raises SystemExit with an actionable message (never a silent partial model)
    when a class is too small, or when a required app category has no images and
    ``allow_partial`` was not requested.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[Path]] = {}
    skipped: dict[str, str] = {}
    source_folder: dict[str, str] = {}

    # Folders are resolved through the declared normalization/aliases, so a
    # mixed tree (TrashNet's "glass" + garbage12's brown-/green-/white-glass)
    # yields four distinct output classes that all stay in "recyclable".
    folder_by_class, unrecognized, ambiguous = resolve_folders(images_dir, mapping)
    if ambiguous:
        raise SystemExit(
            f"Ambiguous dataset folders under {images_dir}: {'; '.join(ambiguous)}. "
            "Keep exactly one folder per model class so no two sources are silently merged."
        )
    for folder_name in unrecognized:
        skipped[folder_name] = "folder name is not in the mapping/alias table (not classified)"

    # Folders that are real mapping keys but hold no usable images are reported
    # rather than becoming an empty (untrained) output class.
    for class_name in mapping:
        if class_name in folder_by_class:
            continue
        folder_path = images_dir / class_name
        if folder_path.is_dir():
            skipped[class_name] = "no usable image files"

    for class_name in sorted(folder_by_class):
        class_dir = folder_by_class[class_name]
        files = _image_files(class_dir)
        if not files:
            skipped[class_name] = "no usable image files"
            continue
        rng.shuffle(files)
        if max_per_class > 0:
            files = files[:max_per_class]  # seeded cap to limit class imbalance (e.g. 5k clothes)
        if len(files) < MIN_IMAGES_PER_CLASS:
            raise SystemExit(
                f"Dataset class '{class_name}' (folder '{class_dir.name}') has only {len(files)} "
                f"usable image(s) in {class_dir}; at least {MIN_IMAGES_PER_CLASS} are needed to build "
                "train/val/test splits. Add more images or remove that class folder."
            )
        by_class[class_name] = files
        source_folder[class_name] = class_dir.name

    if not by_class:
        raise SystemExit(f"No usable class folders with images under {images_dir}")

    classes = sorted(by_class)
    class_to_category = {c: mapping[c] for c in classes}

    # Coverage is checked BEFORE writing anything: a class that is mapped but
    # absent must never be reported as supported.
    try:
        missing_categories = check_category_coverage(
            class_to_category, mapping_tables=(mapping,), allow_partial=allow_partial
        )
    except DatasetCoverageError as e:
        message = str(e)
        if skipped:
            message += f"\nSkipped folders (no usable images): {skipped}"
        message += (
            f"\nDataset folder: {images_dir}\nDetected class folders with images: {classes}"
        )
        raise SystemExit(message) from e

    split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for class_name in classes:
        files = by_class[class_name]
        try:
            n_train, n_val, _ = split_counts(len(files))
        except ManifestError as e:
            raise SystemExit(f"Dataset class '{class_name}': {e}") from e
        bounds = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for subset, subset_files in bounds.items():
            for p in subset_files:
                split[subset].append(
                    {"path": str(p), "dataset_class": class_name, "category": class_to_category[class_name]}
                )

    counts = {
        subset: {
            c: sum(1 for r in rows if r["dataset_class"] == c)
            for c in classes
            if any(r["dataset_class"] == c for r in rows)
        }
        for subset, rows in split.items()
    }
    manifest = {
        "dataset_dir": str(images_dir),
        "dataset_name": images_dir.name,
        "class_to_category": class_to_category,
        "classes": classes,
        "num_classes": len(classes),
        "idx_to_class": {i: c for i, c in enumerate(classes)},
        "seed": int(seed),
        "max_per_class": max_per_class,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "skipped_folders": skipped,
        "coverage": {
            "app_categories": APP_CATEGORIES,
            "present": sorted(set(class_to_category.values())),
            "missing": {c: v for c, v in sorted(missing_categories.items())},
            "partial_training": bool(missing_categories),
        },
        "counts": counts,
        "sizes": {s: len(rows) for s, rows in split.items()},
        **split,  # the actual per-image rows used by train.py / evaluate.py
        "licences": {
            "trashnet": "MIT (c) 2017 Gary Thung — github.com/garythung/trashnet/LICENSE",
            "garbage12": "Kaggle mostafaabla/garbage-classification (12 classes, NO e-waste class) — check licence on Kaggle before redistribution",
            "giz_ewaste": "CC BY 4.0 — huggingface.co/datasets/GIZ/E-Waste-Database (all images are e-waste)",
        },
    }

    # Final contract gate: the manifest is written only if it is trainable and
    # internally consistent (num_classes == classes, no empty output class,
    # no split leakage, every class mapped to a category the app can serve).
    try:
        report = validate_manifest(manifest, mapping_tables=(mapping,), allow_partial=allow_partial)
    except ManifestError as e:
        raise SystemExit(f"Refusing to write an invalid split manifest: {e}") from e
    manifest["coverage"]["partial_training"] = report["partial_training"]
    manifest["sizes"] = report["sizes"]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "split.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"Split written to {path}: train={manifest['sizes']['train']} "
          f"val={manifest['sizes']['val']} test={manifest['sizes']['test']} "
          f"classes={classes} (num_classes={manifest['num_classes']})")
    for subset, subset_counts in manifest["counts"].items():
        print(f"  {subset}: {subset_counts}")
    if skipped:
        print(f"Skipped folders without usable images (not output classes): {skipped}")
    if manifest["coverage"]["missing"]:
        print("WARNING: partial training — these app categories cannot be predicted locally: "
              f"{sorted(manifest['coverage']['missing'])}")
        for category, needed in manifest["coverage"]["missing"].items():
            print(f"  - '{category}' needs dataset class folder(s): {needed or 'any class mapped to it'}")
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
    ap.add_argument("--seed", type=int, default=SEED, help="random seed for the split and training")
    ap.add_argument("--allow-partial", action="store_true",
                    help="explicitly allow training only the categories the dataset covers "
                         "(the manifest records partial_training=true; /health and the app "
                         "then report the reduced coverage). Without this flag a missing "
                         "category aborts preparation with an error.")
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

    mapped_missing = [c for c in CATEGORIES if c != "uncertain" and c not in set(mapping.values())]
    if mapped_missing:
        print(f"NOTE: mapping '{args.source}' never maps {mapped_missing}; those categories stay unreachable "
              "unless --allow-partial is used.")
    build_split(images_dir, mapping, args.out_dir, args.max_per_class,
                allow_partial=args.allow_partial, seed=args.seed)


if __name__ == "__main__":
    main()
