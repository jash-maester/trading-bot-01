"""Device selection and reproducibility utilities."""
from __future__ import annotations

import random

import numpy as np
import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Return the best available torch device.

    Priority: cuda → mps → cpu, unless overridden by `prefer`.
    Never hardcode 'cuda' in training code — call this instead.
    """
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
