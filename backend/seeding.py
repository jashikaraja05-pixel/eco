"""Deterministic seeding shared by train.py, evaluate.py and the tests.

Torch and NumPy are imported lazily so the seeding contract can be unit-tested
in an environment without PyTorch. The returned report is written into
training_summary.json so a run can be reproduced from its own artefacts.
"""
from __future__ import annotations

import os
import random
from typing import Any

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True) -> dict[str, Any]:
    """Seeds Python, NumPy and PyTorch (CPU + CUDA) and reports what was seeded.

    Returns {"seed", "python", "numpy", "torch", "cuda", "deterministic_algorithms",
    "cudnn_benchmark"} — missing libraries are reported as False instead of
    raising, so the same call works in tests without torch installed.
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    report: dict[str, Any] = {
        "seed": seed,
        "python": True,
        "numpy": False,
        "torch": False,
        "cuda": False,
        "deterministic_algorithms": False,
        "cudnn_benchmark": None,
    }

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard requirement in practice
        pass
    else:
        np.random.seed(seed)
        report["numpy"] = True

    try:
        import torch
    except ImportError:
        return report

    torch.manual_seed(seed)
    report["torch"] = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        report["cuda"] = True
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            report["deterministic_algorithms"] = True
        except (AttributeError, RuntimeError):  # pragma: no cover - old torch
            report["deterministic_algorithms"] = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            report["cudnn_benchmark"] = False
    return report


def loader_generator(seed: int, epoch: int = 0):
    """Seeded torch.Generator for DataLoader shuffling (raises without torch)."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch))
    return generator
