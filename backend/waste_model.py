"""Local PyTorch waste classifier (MobileNetV3-Small, transfer learning).

The model is trained on a labelled public waste dataset (see prepare_dataset.py
and backend/README.md). It predicts *dataset* classes; a stored mapping turns
them into the app's six categories. Bins and the uncertainty rule are applied
by common.py / the server, never by the model itself.

No prediction is ever random or hardcoded: if weights or preprocessing config
are missing, the caller (main.py) falls back to nothing — it reports that local
inference is unavailable instead of inventing a result.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from common import UNCERTAIN_THRESHOLD

DEFAULT_MODELS_DIR = Path(__file__).parent / "models"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "best.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small with a new class head.

    pretrained=True downloads ImageNet weights (training only). Inference uses
    pretrained=False and loads the saved checkpoint, so no network is needed.
    """
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def load_model(models_dir: Path | str = DEFAULT_MODELS_DIR) -> tuple[nn.Module, dict[str, Any]]:
    """Loads trained weights + preprocessing config. Raises FileNotFoundError if absent."""
    models_dir = Path(models_dir)
    config = json.loads((models_dir / CONFIG_NAME).read_text())
    weights_path = models_dir / WEIGHTS_NAME
    if not weights_path.exists():
        raise FileNotFoundError(f"No trained weights at {weights_path} — run train.py first.")
    model = build_model(int(config["num_classes"]), pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, config


def preprocess(config: dict[str, Any], train: bool = False) -> transforms.Compose:
    size = int(config.get("image_size", 224))
    if train:
        return transforms.Compose([
            transforms.Resize(int(size * 1.15)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(size * 1.15)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def predict_image(model: nn.Module, config: dict[str, Any], image: Image.Image) -> dict[str, Any]:
    """Runs real local inference. Returns an honest raw prediction dict.

    `item` is the predicted *dataset* class name (the finest label the model
    has) — never an invented product name. Reasoning text states what actually
    happened: a CNN category prediction with a confidence score.
    """
    idx_to_class = {int(k): v for k, v in config["idx_to_class"].items()}
    tensor = preprocess(config)(image.convert("RGB")).unsqueeze(0)
    torch.set_grad_enabled(False)
    start = time.perf_counter()
    logits = model(tensor)
    probs = torch.softmax(logits[0], dim=0)
    elapsed_ms = (time.perf_counter() - start) * 1000
    conf, idx = probs.max(0)
    dataset_class = idx_to_class[int(idx)]
    category = config["class_to_category"][dataset_class]
    confidence = round(float(conf) * 100)
    if confidence < UNCERTAIN_THRESHOLD:
        category = "uncertain"
    return {
        "item": dataset_class,
        "category": category,
        "confidence": confidence,
        "reasoning": (
            f"Local MobileNetV3 classifier (trained on {config.get('dataset_name', 'a public waste dataset')}) "
            f"matched this image to dataset class '{dataset_class}' with {confidence}% confidence; "
            f"confidence below {UNCERTAIN_THRESHOLD}% is reported as uncertain."
        ),
        "_latency_ms": round(elapsed_ms, 1),
    }
