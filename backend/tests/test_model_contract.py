import pytest

from common import BINS, CATEGORIES, UNCERTAIN_THRESHOLD, sniff


def test_bin_mapping_covers_all_categories():
    assert set(BINS) == set(CATEGORIES)
    assert len(BINS) == 6


def test_uncertainty_threshold_matches_frontend():
    # Keep in sync with src/lib/scan-schema.ts UNCERTAIN_THRESHOLD.
    assert UNCERTAIN_THRESHOLD == 55


def test_sniff_detects_real_formats():
    assert sniff(b"\xff\xd8\xff" + b"0" * 32) == "image/jpeg"
    assert sniff(b"\x89PNG\r\n\x1a\n" + b"0" * 32) == "image/png"
    assert sniff(b"RIFF" + b"0" * 4 + b"WEBP" + b"0" * 16) == "image/webp"
    assert sniff(b"GIF89a....") is None
    assert sniff(b"<html>") is None


def test_waste_model_predict_contract():
    pytest.importorskip("torch", reason="torch not installed in this environment")
    import torch
    from PIL import Image

    from waste_model import build_model, predict_image

    config = {
        "num_classes": 6,
        "idx_to_class": {i: c for i, c in enumerate(["cardboard", "glass", "metal", "paper", "plastic", "trash"])},
        "class_to_category": {"cardboard": "recyclable", "glass": "recyclable", "metal": "recyclable",
                              "paper": "recyclable", "plastic": "recyclable", "trash": "residual"},
        "dataset_name": "trashnet",
        "image_size": 224,
    }
    # untrained head is fine for a contract check: shape + honest output fields
    model = build_model(6)
    model.eval()
    image = Image.new("RGB", (224, 224), "green")
    p = predict_image(model, config, image)
    assert p["category"] in CATEGORIES + ["uncertain"]
    assert 0 <= p["confidence"] <= 100
    assert p["item"] in config["idx_to_class"].values()
    assert torch.is_tensor(torch.zeros(1))
