"""Dataset split-manifest contract shared by the EcoLens ML scripts.

This module is deliberately torch-free so that the dataset/training contract can
be validated and unit-tested without PyTorch installed. It is the single place
that decides:

* which dataset classes become model output classes — only classes that
  actually contain usable images, never a mapping key whose folder is missing
  or empty (that was the cause of an untrained output neuron: 13 mapping keys
  vs 12 real folders);
* whether the dataset can genuinely serve the app's five waste categories, and
  which dataset class folder is required for each missing category;
* that ``classes``, ``num_classes``, ``idx_to_class`` and ``class_to_category``
  agree, that no output class is empty, and that no image appears in two splits.

Nothing here fabricates results: an unusable manifest raises ``ManifestError``
(or ``DatasetCoverageError`` for missing categories) instead of being trained.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from common import CATEGORIES

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Categories the classifier must be able to predict. "uncertain" is produced by
# the confidence threshold, never by a dataset class.
APP_CATEGORIES = [c for c in CATEGORIES if c != "uncertain"]

# A class below this many usable images cannot be split into train/val/test.
MIN_IMAGES_PER_CLASS = 9

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # test gets the remainder
DEFAULT_SEED = 42


class ManifestError(ValueError):
    """The split manifest is internally inconsistent or unusable for training."""


class DatasetCoverageError(ManifestError):
    """The dataset cannot train one or more of the app's five categories."""

    def __init__(self, missing: Mapping[str, Sequence[str]], found_classes: Sequence[str] = (),
                 partial_hint: bool = True):
        self.missing = {str(k): list(v) for k, v in missing.items()}
        self.found_classes = list(found_classes)
        lines = ["Dataset cannot train every waste category the app serves:"]
        for category, source_classes in self.missing.items():
            required = ", ".join(f"'{c}'" for c in source_classes) or "a class mapped to this category"
            lines.append(f"  - category '{category}' has no usable images; required dataset class: {required}")
        lines.append(f"Missing categories: {sorted(self.missing)}")
        if self.found_classes:
            lines.append(f"Dataset classes with usable images: {sorted(self.found_classes)}")
        if partial_hint:
            lines.append(
                "Nothing was trained or written for those categories. Add the class folders above, or pass "
                "--allow-partial to train only the covered categories — the manifest then records "
                "partial_training=true and the reduced coverage is reported by /health."
            )
        super().__init__("\n".join(lines))


def category_source_classes(mapping_tables: Iterable[Mapping[str, str]] = ()) -> dict[str, list[str]]:
    """Reverse lookup category -> dataset class names that provide it."""
    out: defaultdict[str, list[str]] = defaultdict(list)
    for table in mapping_tables:
        for dataset_class, category in table.items():
            if dataset_class not in out[category]:
                out[category].append(dataset_class)
    return {category: sorted(classes) for category, classes in out.items()}


def required_dataset_classes(category: str, mapping_tables: Iterable[Mapping[str, str]] = ()) -> list[str]:
    return category_source_classes(mapping_tables).get(category, [])


NO_SOURCE_HINT = "any class mapped to this category (no mapping entry exists yet)"


def check_category_coverage(class_to_category: Mapping[str, str],
                            mapping_tables: Iterable[Mapping[str, str]] = (),
                            allow_partial: bool = False) -> dict[str, list[str]]:
    """Returns {missing category: [required dataset classes]}; raises unless partial is allowed."""
    tables = tuple(mapping_tables)
    covered = set(class_to_category.values())
    missing = {
        category: required_dataset_classes(category, tables)
        for category in APP_CATEGORIES
        if category not in covered
    }
    if missing and not allow_partial:
        raise DatasetCoverageError(missing, found_classes=sorted(class_to_category))
    return missing


def split_counts(n: int) -> tuple[int, int, int]:
    """(train, val, test) sizes for n images, guaranteeing >=1 row in each split."""
    if n < 3:
        raise ManifestError(f"A class needs at least 3 images to be split into train/val/test, got {n}.")
    n_val = max(1, round(n * VAL_FRAC))
    n_test = max(1, round(n * (1.0 - TRAIN_FRAC - VAL_FRAC)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ManifestError(
            f"A class needs at least 3 images to keep a training row, got {n} "
            f"(would leave train={n_train}, val={n_val}, test={n_test})."
        )
    return n_train, n_val, n_test


def normalize_idx_to_class(raw: Mapping[Any, str]) -> dict[int, str]:
    """JSON turns int keys into strings; normalize back to int -> class."""
    try:
        return {int(k): str(v) for k, v in raw.items()}
    except (TypeError, ValueError) as e:
        raise ManifestError(f"idx_to_class has a non-integer key: {raw!r}") from e


def _paths_equal(a: str, b: str) -> bool:  # pragma: no cover - helper kept for clarity
    return Path(a) == Path(b)


def validate_manifest(manifest: Mapping[str, Any], *,
                      mapping_tables: Iterable[Mapping[str, str]] = (),
                      allow_partial: bool = False,
                      min_train_per_class: int = 1) -> dict[str, Any]:
    """Validates a split manifest. Raises ManifestError / DatasetCoverageError.

    Returns a report describing the accepted contract (class count, split sizes,
    train rows per class, category coverage and whether training is partial).
    """
    if not isinstance(manifest, Mapping):
        raise ManifestError("Split manifest must be a JSON object.")
    for key in ("classes", "class_to_category", "train", "val", "test"):
        if key not in manifest:
            raise ManifestError(f"Split manifest is missing the required key '{key}'.")

    classes = manifest["classes"]
    if not isinstance(classes, list) or not classes:
        raise ManifestError("Manifest 'classes' must be a non-empty list of dataset class names.")
    if any(not isinstance(c, str) or not c.strip() for c in classes):
        raise ManifestError("Manifest 'classes' must only contain non-empty strings.")
    duplicate_classes = sorted({c for c in classes if classes.count(c) > 1})
    if duplicate_classes:
        raise ManifestError(f"Manifest 'classes' contains duplicates: {duplicate_classes}")
    class_set = set(classes)

    raw_map = manifest["class_to_category"]
    if not isinstance(raw_map, Mapping) or not raw_map:
        raise ManifestError("Manifest 'class_to_category' must be a non-empty mapping.")
    class_to_category = {str(k): str(v) for k, v in raw_map.items()}
    missing_map = sorted(class_set - set(class_to_category))
    extra_map = sorted(set(class_to_category) - class_set)
    if missing_map or extra_map:
        raise ManifestError(
            "class_to_category is inconsistent with the manifest classes: "
            f"missing {missing_map or 'none'}, unexpected {extra_map or 'none'}. "
            "Every output class needs exactly one category, and only output classes may be mapped."
        )
    bad_categories = sorted({v for v in class_to_category.values() if v not in APP_CATEGORIES})
    if bad_categories:
        raise ManifestError(
            f"class_to_category maps classes to categories the app cannot serve: {bad_categories}. "
            f"Allowed dataset categories: {APP_CATEGORIES} ('uncertain' is never a dataset class)."
        )

    num_classes = manifest.get("num_classes")
    if num_classes is not None and int(num_classes) != len(classes):
        raise ManifestError(
            f"Manifest num_classes={num_classes} does not match its {len(classes)} classes {classes}. "
            "An extra output neuron would be randomly initialised and never trained."
        )
    if manifest.get("idx_to_class"):
        idx_to_class = normalize_idx_to_class(manifest["idx_to_class"])
        ordered = [idx_to_class.get(i) for i in range(len(classes))]
        if sorted(idx_to_class) != list(range(len(classes))) or ordered != classes:
            raise ManifestError(
                f"Manifest idx_to_class {dict(sorted(idx_to_class.items()))} is inconsistent with classes {classes}."
            )

    sizes: Counter[str] = Counter()
    train_counts: Counter[str] = Counter()
    seen_paths: dict[str, str] = {}
    used_classes: set[str] = set()
    for subset in ("train", "val", "test"):
        rows = manifest[subset]
        if not isinstance(rows, list):
            raise ManifestError(f"Manifest '{subset}' must be a list of image rows.")
        for row in rows:
            if not isinstance(row, Mapping) or "path" not in row or "dataset_class" not in row:
                raise ManifestError(f"Manifest '{subset}' row is missing 'path'/'dataset_class': {row!r}")
            dataset_class = str(row["dataset_class"])
            if dataset_class not in class_set:
                raise ManifestError(
                    f"Manifest '{subset}' row uses dataset_class '{dataset_class}', "
                    f"which is not one of the manifest classes {classes}."
                )
            if "category" in row and str(row["category"]) != class_to_category[dataset_class]:
                raise ManifestError(
                    f"Manifest '{subset}' row for '{dataset_class}' says category '{row['category']}' "
                    f"but class_to_category says '{class_to_category[dataset_class]}'."
                )
            key = Path(str(row["path"])).resolve().as_posix()
            if key in seen_paths:
                previous = seen_paths[key]
                if previous == subset:
                    raise ManifestError(f"Duplicate image row inside '{subset}': {row['path']}")
                raise ManifestError(
                    f"Split leakage: image {row['path']} appears in both '{previous}' and '{subset}'."
                )
            seen_paths[key] = subset
            sizes[subset] += 1
            used_classes.add(dataset_class)
            if subset == "train":
                train_counts[dataset_class] += 1

    if not sizes["train"]:
        raise ManifestError("Manifest has no training rows.")

    empty_classes = sorted(class_set - set(train_counts))
    if empty_classes:
        raise ManifestError(
            f"Empty output classes: {empty_classes} have no training images. "
            "Every class in 'classes' must be learnable — drop the class or add its images."
        )
    thin_classes = {c: n for c, n in sorted(train_counts.items()) if n < min_train_per_class}
    if thin_classes:
        raise ManifestError(
            f"Classes with fewer than {min_train_per_class} training images: {thin_classes}."
        )
    unused_classes = sorted(class_set - used_classes)
    if unused_classes:
        raise ManifestError(f"Manifest classes with no images in any split: {unused_classes}.")

    declared_counts = manifest.get("counts")
    if isinstance(declared_counts, Mapping):
        for subset in ("train", "val", "test"):
            claimed = declared_counts.get(subset) or {}
            actual = Counter(str(str(r["dataset_class"])) for r in manifest[subset])
            declared = {str(k): int(v) for k, v in dict(claimed).items()}
            if {k: v for k, v in declared.items() if v} != {k: v for k, v in actual.items() if v}:
                raise ManifestError(
                    f"Manifest 'counts.{subset}' {declared} does not match its actual rows {dict(actual)}."
                )
    declared_sizes = manifest.get("sizes")
    if isinstance(declared_sizes, Mapping):
        for subset in ("train", "val", "test"):
            if subset in declared_sizes and int(declared_sizes[subset]) != sizes[subset]:
                raise ManifestError(
                    f"Manifest 'sizes.{subset}'={declared_sizes[subset]} but the split has {sizes[subset]} rows."
                )

    # check_category_coverage raises unless partial training was allowed, so a
    # returned non-empty mapping means this run really trains on reduced
    # coverage — and that fact must be recorded, not inferred from a stale flag.
    missing_categories = check_category_coverage(class_to_category, mapping_tables, allow_partial=allow_partial)
    partial_training = bool(missing_categories)

    return {
        "num_classes": len(classes),
        "classes": list(classes),
        "class_to_category": dict(sorted(class_to_category.items())),
        "idx_to_class": {i: c for i, c in enumerate(classes)},
        "sizes": {s: sizes[s] for s in ("train", "val", "test")},
        "train_counts": dict(sorted(train_counts.items())),
        "present_categories": sorted(set(class_to_category.values())),
        "missing_categories": missing_categories,
        "partial_training": partial_training,
        "images": sum(sizes.values()),
    }


def build_model_config(manifest: Mapping[str, Any], *,
                       mapping_tables: Iterable[Mapping[str, str]] = (),
                       allow_partial: bool = False,
                       image_size: int = 224,
                       uncertain_threshold: int = 55,
                       seed: int | None = None,
                       dataset_name: str | None = None,
                       normalization: Mapping[str, Sequence[float]] | None = None) -> dict[str, Any]:
    """Builds the model config written next to best.pt from a validated manifest.

    num_classes, classes, idx_to_class and class_to_category are derived from one
    validated source, so they cannot disagree with the checkpoint's output head.
    """
    report = validate_manifest(manifest, mapping_tables=mapping_tables, allow_partial=allow_partial)
    config: dict[str, Any] = {
        "num_classes": report["num_classes"],
        "classes": report["classes"],
        "idx_to_class": report["idx_to_class"],
        "class_to_category": report["class_to_category"],
        "image_size": int(image_size),
        "uncertain_threshold": int(uncertain_threshold),
        "dataset_name": dataset_name or Path(str(manifest.get("dataset_dir", "dataset"))).name,
        "dataset_licences": manifest.get("licences", {}),
        "split_sizes": report["sizes"],
        "category_coverage": report["present_categories"],
        "missing_categories": sorted(report["missing_categories"]),
        "partial_training": report["partial_training"],
        "seed": int(seed if seed is not None else manifest.get("seed", DEFAULT_SEED)),
    }
    if normalization is not None:
        config["normalization"] = {k: list(v) for k, v in normalization.items()}
    return config
