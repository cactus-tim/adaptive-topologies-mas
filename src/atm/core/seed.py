"""Reproducibility helper — seed all RNGs."""

from __future__ import annotations

import os
import random


def seed_all(seed: int) -> None:
    """Seed Python's random, numpy (if installed), and torch (if installed).

    Args:
        seed: Non-negative integer seed.

    Idempotent — safe to call multiple times.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
    except ImportError:
        pass
