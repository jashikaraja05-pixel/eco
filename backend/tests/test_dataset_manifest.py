"""Contract tests for the dataset mapping / manifest / training gate.

These tests are torch-free: they build tiny synthetic class folders on disk (no
image decoding is needed — prepare_dataset only checks extension and size) and
assert that the code refuses to produce a class the dataset cannot train. This
is the regression guard for the original bug: a 13-key GARBAGE12 mapping over a
12-folder dataset used to yield num_classes=13 with one permanently untrained
output neuron.

No dataset is downloaded and no model is trained here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from manifest import (
    APP_CATEGORIES,
    DatasetCoverageError,
    ManifestError,
    check_category_coverage,
    required_dataset_classes,
    split_counts,
    validate_manifest,
)
from prepare_dataset import (
    GARBAGE12_MAPPING,
    MERGED_MAPPING,
    TRASHNET_MAPPING,
    build_split,
    class_folders,
    normalize_class_name,
    resolve_folders,
)
from seeding import seed_everything

FIVE_CATEGORY_MAPPING = {
    "battery": "hazardous",
    "biological": "organic",
    "cardboard": "recyclable",
    "e-waste": "ewaste",
    "trash": "residual",
}


def make_class_dir(root: Path, name: str, n: int) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"{name}_{i:03d}.jpg").write_bytes(b"\xff\xd8\xff" + bytes([i % 250]) * 8)
    return d


def build(tmp_path: Path, mapping: dict[str, str], counts: dict[str, int], **kwargs) -> dict:
    root = tmp_path / "dataset"
    for name, n in counts.items():
        if n > 0:
            make_class_dir(root, name, n)
        else:
            (root / name).mkdir(parents=True, exist_ok=True)  # present but empty
    path = build_split(root, mapping, tmp_path / "out", **kwargs)
    return json.loads(path.read_text())


# --- TASK 1: classes come only from folders that contain usable images --------

def test_missing_mapped_folder_never_becomes_an_output_class(tmp_path):
    """GARBAGE12 maps 13 keys but the Kaggle archive has only 12 folders."""
    counts = {c: 12 for c in GARBAGE12_MAPPING if c != "e-waste"}
    manifest = build(tmp_path, GARBAGE12_MAPPING, counts, allow_partial=True)

    assert "e-waste" not in manifest["classes"]
    assert "ewaste" not in set(manifest["class_to_category"].values()) or "e-waste" in manifest["classes"]
    assert manifest["num_classes"] == len(manifest["classes"]) == 12
    assert set(manifest["idx_to_class"].values()) == set(manifest["classes"])
    # the cause of the bug: no output class may exist without images in every split
    for subset in ("train", "val", "test"):
        assert set(r["dataset_class"] for r in manifest[subset]) == set(manifest["classes"])


def test_empty_folder_is_skipped_and_reported(tmp_path):
    counts = {"battery": 12, "biological": 12, "cardboard": 12, "trash": 12, "e-waste": 0}
    manifest = build(tmp_path, FIVE_CATEGORY_MAPPING, counts, allow_partial=True)

    assert "e-waste" not in manifest["classes"]
    assert manifest["skipped_folders"]["e-waste"] == "no usable image files"
    assert "ewaste" in manifest["coverage"]["missing"]


def test_class_below_minimum_is_rejected(tmp_path):
    counts = {"battery": 12, "biological": 12, "cardboard": 12, "trash": 12, "e-waste": 2}
    with pytest.raises(SystemExit) as exc:
        build(tmp_path, FIVE_CATEGORY_MAPPING, counts, allow_partial=True)
    assert "e-waste" in str(exc.value)


# --- category coverage -------------------------------------------------------

def test_required_dataset_classes_lists_every_source_class():
    assert required_dataset_classes("ewaste", (GARBAGE12_MAPPING,)) == ["e-waste"]
    assert required_dataset_classes("organic", (GARBAGE12_MAPPING,)) == ["biological"]
    assert required_dataset_classes("recyclable", (TRASHNET_MAPPING,)) == [
        "cardboard", "glass", "metal", "paper", "plastic"
    ]


def test_missing_category_aborts_with_actionable_error(tmp_path):
    """trashnet has no organic/hazardous/e-waste images: preparation must stop."""
    counts = {c: 12 for c in TRASHNET_MAPPING}
    with pytest.raises(SystemExit) as exc:
        build(tmp_path, TRASHNET_MAPPING, counts)
    message = str(exc.value)
    assert "organic" in message and "hazardous" in message and "ewaste" in message
    for required in ("biological", "battery", "e-waste"):
        assert required in message
    assert "--allow-partial" in message
    assert not (tmp_path / "out" / "split.json").exists()


def test_partial_training_is_explicit_and_recorded(tmp_path):
    counts = {c: 12 for c in TRASHNET_MAPPING}
    manifest = build(tmp_path, TRASHNET_MAPPING, counts, allow_partial=True)

    assert manifest["coverage"]["partial_training"] is True
    assert sorted(manifest["coverage"]["missing"]) == ["ewaste", "hazardous", "organic"]
    assert manifest["coverage"]["present"] == ["recyclable", "residual"]
    # and the contract accepts it only when partial training is restated
    report = validate_manifest(manifest, mapping_tables=(TRASHNET_MAPPING,), allow_partial=True)
    assert report["partial_training"] is True
    assert sorted(report["missing_categories"]) == ["ewaste", "hazardous", "organic"]
    with pytest.raises(DatasetCoverageError):
        validate_manifest(manifest, mapping_tables=(TRASHNET_MAPPING,), allow_partial=False)


def test_full_coverage_is_not_flagged_partial(tmp_path):
    counts = {c: 12 for c in FIVE_CATEGORY_MAPPING}
    manifest = build(tmp_path, FIVE_CATEGORY_MAPPING, counts)

    assert manifest["coverage"]["partial_training"] is False
    assert manifest["coverage"]["missing"] == {}
    report = validate_manifest(manifest, mapping_tables=(FIVE_CATEGORY_MAPPING,))
    assert set(report["present_categories"]) == set(APP_CATEGORIES)
    assert report["partial_training"] is False


def test_category_coverage_error_names_category_and_classes():
    with pytest.raises(DatasetCoverageError) as exc:
        check_category_coverage({"cardboard": "recyclable"}, (GARBAGE12_MAPPING,))
    err = exc.value
    assert "organic" in err.missing and "ewaste" in err.missing
    assert err.missing["ewaste"] == ["e-waste"]
    assert "biological" in str(err)


# --- TASK 2: training contract ----------------------------------------------

def test_num_classes_must_equal_number_of_classes(tmp_path):
    counts = {c: 12 for c in FIVE_CATEGORY_MAPPING}
    manifest = build(tmp_path, FIVE_CATEGORY_MAPPING, counts)
    manifest["num_classes"] += 1
    with pytest.raises(ManifestError) as exc:
        validate_manifest(manifest, allow_partial=True)
    assert "num_classes" in str(exc.value)


def test_output_class_without_training_images_is_rejected(tmp_path):
    counts = {c: 12 for c in FIVE_CATEGORY_MAPPING}
    manifest = build(tmp_path, FIVE_CATEGORY_MAPPING, counts)
    manifest["train"] = [r for r in manifest["train"] if r["dataset_class"] != "battery"]
    manifest.pop("counts", None)
    manifest.pop("sizes", None)
    with pytest.raises(ManifestError) as exc:
        validate_manifest(manifest, allow_partial=True)
    assert "battery" in str(exc.value)


def test_class_mapping_must_cover_exactly_the_classes(tmp_path):
    counts = {c: 12 for c in FIVE_CATEGORY_MAPPING}
    manifest = build(tmp_path, FIVE_CATEGORY_MAPPING, counts)

    missing = json.loads(json.dumps(manifest))
    missing["class_to_category"].pop("battery")
    with pytest.raises(ManifestError):
        validate_manifest(missing, allow_partial=True)

    extra = json.loads(json.dumps(manifest))
    extra["class_to_category"]["not-a-class"] = "organic"
    with pytest.raises(ManifestError):
        validate_manifest(extra, allow_partial=True)

    bad_category = json.loads(json.dumps(manifest))
    bad_category["class_to_category"]["battery"] = "uncertain"
    with pytest.raises(ManifestError) as exc:
        validate_manifest(bad_category, allow_partial=True)
    assert "uncertain" in str(exc.value)


def test_idx_to_class_must_match_classes(tmp_path):
    counts = {c: 12 for c in FIVE_CATEGORY_MAPPING}
    manifest = build(tmp_path, FIVE_CATEGORY_MAPPING, counts)

    swapped = json.loads(json.dumps(manifest))
    a, b = swapped["classes"][0], swapped["classes"][1]
    swapped["idx_to_class"]["0"], swapped["idx_to_class"]["1"] = b, a
    with pytest.raises(ManifestError):
        validate_manifest(swapped, allow_partial=True)

    unknown = json.loads(json.dumps(manifest))
    unknown["test"][0]["dataset_class"] = "mystery"
    with pytest.raises(ManifestError) as exc:
        validate_manifest(unknown, allow_partial=True)
    assert "mystery" in str(exc.value)


def test_row_category_must_match_class_mapping(tmp_path):
    counts = {c: 12 for c in FIVE_CATEGORY_MAPPING}
    manifest = build(tmp_path, FIVE_CATEGORY_MAPPING, counts)
    manifest["val"][0]["category"] = "organic"
    with pytest.raises(ManifestError) as exc:
        validate_manifest(manifest, allow_partial=True)
    assert "category" in str(exc.value)


def test_split_leakage_is_rejected(tmp_path):
    counts = {c: 12 for c in FIVE_CATEGORY_MAPPING}
    manifest = build(tmp_path, FIVE_CATEGORY_MAPPING, counts)
    manifest["test"].append(dict(manifest["train"][0]))
    manifest.pop("counts", None)
    manifest.pop("sizes", None)
    with pytest.raises(ManifestError) as exc:
        validate_manifest(manifest, allow_partial=True)
    assert "leakage" in str(exc.value).lower()


def test_declared_counts_must_match_rows(tmp_path):
    counts = {c: 12 for c in FIVE_CATEGORY_MAPPING}
    manifest = build(tmp_path, FIVE_CATEGORY_MAPPING, counts)
    manifest["counts"]["train"]["battery"] += 5
    with pytest.raises(ManifestError) as exc:
        validate_manifest(manifest, allow_partial=True)
    assert "counts" in str(exc.value)


def test_every_split_is_non_empty_and_ordered(tmp_path):
    counts = {c: 12 for c in FIVE_CATEGORY_MAPPING}
    manifest = build(tmp_path, FIVE_CATEGORY_MAPPING, counts)
    report = validate_manifest(manifest, allow_partial=True)
    assert all(report["sizes"][s] > 0 for s in ("train", "val", "test"))
    assert report["classes"] == sorted(manifest["classes"])
    assert report["num_classes"] == 5
    assert report["images"] == sum(report["sizes"].values())


def test_split_counts_keep_a_training_row():
    # 70/15/15 with a guaranteed row in every split.
    assert split_counts(9) == (7, 1, 1)
    assert split_counts(20) == (14, 3, 3)
    for n in range(3, 60):
        train, val, test = split_counts(n)
        assert train >= 1 and val >= 1 and test >= 1 and train + val + test == n
    with pytest.raises(ManifestError):
        split_counts(2)


# --- TASK 3: reproducibility -------------------------------------------------

def test_split_is_reproducible_from_the_seed(tmp_path):
    counts = {c: 20 for c in FIVE_CATEGORY_MAPPING}
    first = build(tmp_path / "a", FIVE_CATEGORY_MAPPING, counts, seed=7)
    second = build(tmp_path / "b", FIVE_CATEGORY_MAPPING, counts, seed=7)
    assert first["seed"] == 7
    assert [r["path"].split("dataset/")[-1] for r in first["train"]] == \
           [r["path"].split("dataset/")[-1] for r in second["train"]]
    # a different seed reshuffles the split (deterministic for these fixed names)
    other = build(tmp_path / "c", FIVE_CATEGORY_MAPPING, counts, seed=8)
    assert [r["path"].split("dataset/")[-1] for r in other["train"]] != \
           [r["path"].split("dataset/")[-1] for r in first["train"]]


def test_seed_everything_reports_and_reproduces():
    import random

    report = seed_everything(1234)
    assert report["seed"] == 1234 and report["python"] is True
    first = [random.random() for _ in range(3)]
    seed_everything(1234)
    assert [random.random() for _ in range(3)] == first

    np = pytest.importorskip("numpy", reason="numpy not installed in this environment")
    seed_everything(99)
    a = np.random.rand(3).tolist()
    seed_everything(99)
    assert np.random.rand(3).tolist() == a


# --- TASK 2A: merged TrashNet + garbage12 folder names -----------------------

def test_merged_mapping_preserves_every_category():
    """MERGED_MAPPING is a union: no key or category may change."""
    for table in (TRASHNET_MAPPING, GARBAGE12_MAPPING):
        for name, category in table.items():
            assert MERGED_MAPPING[name] == category
    assert set(MERGED_MAPPING) == set(TRASHNET_MAPPING) | set(GARBAGE12_MAPPING)
    assert set(MERGED_MAPPING.values()) == set(TRASHNET_MAPPING.values()) | set(GARBAGE12_MAPPING.values())
    # the previously-colliding key: TrashNet's single glass folder stays recyclable
    assert MERGED_MAPPING["glass"] == "recyclable"
    for name in ("brown-glass", "green-glass", "white-glass"):
        assert MERGED_MAPPING[name] == "recyclable"


def test_normalization_is_case_and_separator_insensitive():
    assert normalize_class_name("E-Waste") == normalize_class_name("e_waste") == normalize_class_name("e-waste")
    assert normalize_class_name(" Brown Glass ") == normalize_class_name("brown-glass")
    # normalization must not merge two genuinely different classes
    assert normalize_class_name("glass") != normalize_class_name("brown-glass")


def test_merged_thirteen_folder_tree_is_recognised(tmp_path):
    """A merged 13-folder tree (12 garbage12 + TrashNet's glass + e-waste)."""
    root = tmp_path / "dataset"
    for name in MERGED_MAPPING:
        make_class_dir(root, name, 12)

    resolved_dir, mapping = class_folders(root)

    assert resolved_dir == root and mapping is MERGED_MAPPING
    folder_by_class, unrecognized, ambiguous = resolve_folders(root, mapping)
    assert unrecognized == [] and ambiguous == []
    # 14 folders on disk, 14 distinct model classes: "glass" is NOT folded into
    # brown-/green-/white-glass, so its images keep their own output neuron.
    assert len([p for p in root.iterdir() if p.is_dir()]) == len(MERGED_MAPPING)
    assert set(folder_by_class) == set(MERGED_MAPPING)
    assert folder_by_class["glass"].name == "glass"
    manifest_classes = json.loads(
        build_split(root, mapping, tmp_path / "out", allow_partial=True).read_text()
    )
    assert len(manifest_classes["classes"]) == len(MERGED_MAPPING)
    assert all(manifest_classes["class_to_category"][c] == "recyclable"
               for c in ("glass", "brown-glass", "green-glass", "white-glass"))
    assert "e-waste" in manifest_classes["classes"]
    assert manifest_classes["class_to_category"]["e-waste"] == "ewaste"


def test_dotted_or_spaced_folder_names_resolve_to_the_same_class(tmp_path):
    root = tmp_path / "dataset"
    make_class_dir(root, "E-Waste", 12)
    make_class_dir(root, "Brown Glass", 12)
    make_class_dir(root, "biological", 12)

    folder_by_class, unrecognized, ambiguous = resolve_folders(root, MERGED_MAPPING)
    assert unrecognized == [] and ambiguous == []
    assert folder_by_class["e-waste"].name == "E-Waste"
    assert folder_by_class["brown-glass"].name == "Brown Glass"


def test_unknown_folder_is_never_silently_classified(tmp_path):
    root = tmp_path / "dataset"
    for name in ("biological", "cardboard", "e-waste"):
        make_class_dir(root, name, 12)
    make_class_dir(root, "asbestos", 12)  # not in any mapping or alias table

    folder_by_class, unrecognized, ambiguous = resolve_folders(root, MERGED_MAPPING)
    assert unrecognized == ["asbestos"]
    assert "asbestos" not in folder_by_class
    assert set(folder_by_class.values()) == {root / "biological", root / "cardboard", root / "e-waste"}

    with pytest.raises(SystemExit) as excinfo:
        class_folders(root)
    assert "asbestos" in str(excinfo.value)


def test_unrecognized_folder_is_skipped_and_reported_in_the_manifest(tmp_path):
    root = tmp_path / "dataset"
    for name in ("biological", "cardboard", "e-waste", "trash"):
        make_class_dir(root, name, 12)
    make_class_dir(root, "asbestos", 12)

    manifest = json.loads(
        build_split(root, MERGED_MAPPING, tmp_path / "out", allow_partial=True).read_text()
    )
    assert "asbestos" not in manifest["classes"]
    assert "asbestos" in manifest["skipped_folders"]
    assert "not in the mapping/alias table" in manifest["skipped_folders"]["asbestos"]


def test_two_folders_for_one_class_are_rejected(tmp_path):
    """'glass' and 'Glass' would silently merge two sources — refuse instead."""
    root = tmp_path / "dataset"
    make_class_dir(root, "glass", 12)
    make_class_dir(root, "Glass", 12)

    with pytest.raises(SystemExit) as excinfo:
        class_folders(root)
    assert "Ambiguous" in str(excinfo.value)
