"""Torch device selection shared by ML services."""

from __future__ import annotations

import torch
from loguru import logger

from ..config import Settings


def resolve_torch_device(settings: Settings, *, warn_on_cpu: bool = True) -> str:
    """Resolve the configured torch device to an available runtime device."""
    if settings.device == "mps" and torch.backends.mps.is_available():
        return "mps"

    if settings.device == "cuda" and torch.cuda.is_available():
        return "cuda"

    if warn_on_cpu:
        logger.warning("MPS/CUDA not available, using CPU")
    return "cpu"
