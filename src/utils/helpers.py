"""
Shared Helper Utilities
========================
Config loading, logging setup, seed management, and miscellaneous functions
used across the project.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a dict.

    Parameters
    ----------
    path : str
        Path to the ``.yaml`` file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy, random, and PyTorch.

    Parameters
    ----------
    seed : int
        Global random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    """
    Configure root logger with a console handler and optional file handler.

    Parameters
    ----------
    level : int
        Logging level (default ``logging.INFO``).
    log_file : str or None
        If provided, also write logs to this file.
    """
    fmt = "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def count_parameters(model) -> int:
    """
    Count the total and trainable parameters of a PyTorch model.

    Returns
    -------
    int
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================
# DEAP / SEED Electrode Positions (3D stereotactic coords)
# ============================================================

# Standard 10-20 system 32-channel positions used in DEAP.
# Coordinates are (x, y, z) in a unit-sphere head model.
DEAP_ELECTRODE_NAMES: list[str] = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "F4", "F8", "FC6", "FC2", "C4", "T8",
    "CP6", "CP2", "P4", "P8", "PO4", "O2", "Fz", "Cz",
]

# Approximate 2D positions (x, y) for topographic plots
DEAP_ELECTRODE_POS_2D: Dict[str, tuple] = {
    "Fp1": (-0.31, 0.95), "AF3": (-0.25, 0.82), "F3": (-0.45, 0.59),
    "F7": (-0.81, 0.59), "FC5": (-0.70, 0.35), "FC1": (-0.25, 0.35),
    "C3": (-0.55, 0.00), "T7": (-0.99, 0.00), "CP5": (-0.70, -0.35),
    "CP1": (-0.25, -0.35), "P3": (-0.45, -0.59), "P7": (-0.81, -0.59),
    "PO3": (-0.25, -0.82), "O1": (-0.18, -0.95), "Oz": (0.00, -0.99),
    "Pz": (0.00, -0.59), "Fp2": (0.31, 0.95), "AF4": (0.25, 0.82),
    "F4": (0.45, 0.59), "F8": (0.81, 0.59), "FC6": (0.70, 0.35),
    "FC2": (0.25, 0.35), "C4": (0.55, 0.00), "T8": (0.99, 0.00),
    "CP6": (0.70, -0.35), "CP2": (0.25, -0.35), "P4": (0.45, -0.59),
    "P8": (0.81, -0.59), "PO4": (0.25, -0.82), "O2": (0.18, -0.95),
    "Fz": (0.00, 0.59), "Cz": (0.00, 0.00),
}

# Frequency band definitions (Hz)
FREQ_BANDS: Dict[str, tuple] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}
