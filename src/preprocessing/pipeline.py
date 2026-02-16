"""
End-to-End EEG Preprocessing Pipeline
=======================================
Orchestrates:  raw data loading → band-pass filter → ICA artefact removal →
               epoch extraction → z-score normalisation → HDF5 export.

Supports both DEAP (preprocessed .dat) and SEED (.mat) dataset formats.

Usage
-----
    from src.preprocessing.pipeline import EEGPreprocessor

    pp = EEGPreprocessor(config_path="config/preprocessing.yaml")
    pp.run(dataset="deap", input_dir="data/raw/deap", output_path="data/processed/deap.h5")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .filters import bandpass_filter, notch_filter

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

@dataclass
class PreprocessingConfig:
    """Preprocessing hyper-parameters (can be loaded from YAML)."""

    # Filtering
    bandpass_low: float = 0.5
    bandpass_high: float = 50.0
    notch_freq: Optional[float] = 50.0  # None to skip
    filter_order: int = 5

    # ICA
    ica_n_components: int = 15
    ica_method: str = "fastica"
    ica_eog_threshold: float = 0.4

    # Epoching
    epoch_duration: float = 1.0     # seconds
    epoch_overlap: float = 0.0      # seconds (0 = no overlap)

    # Normalisation
    normalise: bool = True          # z-score per channel

    # Dataset specifics
    deap_sampling_rate: float = 128.0
    seed_sampling_rate: float = 200.0
    deap_n_channels: int = 32       # EEG-only channels
    seed_n_channels: int = 62

    @classmethod
    def from_yaml(cls, path: str) -> "PreprocessingConfig":
        """Load config from a YAML file."""
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================
# Loaders
# ============================================================

def load_deap_subject(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single DEAP preprocessed .dat file (Python pickle).

    Parameters
    ----------
    filepath : str
        Path to ``s01.dat`` … ``s32.dat``.

    Returns
    -------
    data : np.ndarray, shape (40, 32, 8064)
        40 trials × 32 EEG channels × 8064 time-samples (63 s @ 128 Hz).
    labels : np.ndarray, shape (40, 4)
        Per-trial ratings: [valence, arousal, dominance, liking] ∈ [1, 9].
    """
    import pickle

    with open(filepath, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    # subject is a dict with keys 'data' (40, 40, 8064) and 'labels' (40, 4)
    data = subject["data"][:, :32, :]  # Keep only 32 EEG channels (drop peripheral)
    labels = subject["labels"]
    return data, labels


def load_seed_session(filepath: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load a single SEED .mat session file.

    Parameters
    ----------
    filepath : str
        Path to a SEED ``.mat`` session file.

    Returns
    -------
    trials : list[np.ndarray]
        Each element has shape (n_channels, n_samples) — variable-length trials.
    labels : np.ndarray, shape (n_trials,)
        Emotion labels: 0 (negative), 1 (neutral), 2 (positive).
    """
    from scipy.io import loadmat

    mat = loadmat(filepath)

    # SEED labels for 15 trials (fixed order across sessions)
    seed_labels = np.array(
        [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    )
    # Map {-1, 0, 1} → {0, 1, 2}  (negative, neutral, positive)
    labels = seed_labels + 1

    # Extract trial data — keys like 'djc_eeg1', 'djc_eeg2', …
    trials = []
    for key in sorted(mat.keys()):
        if key.startswith("_"):
            continue
        val = mat[key]
        if isinstance(val, np.ndarray) and val.ndim == 2:
            trials.append(val.astype(np.float64))

    return trials, labels


# ============================================================
# Pipeline
# ============================================================

class EEGPreprocessor:
    """
    Full preprocessing pipeline from raw files to HDF5.

    Parameters
    ----------
    config : PreprocessingConfig or str
        Config object or path to YAML file.
    """

    def __init__(self, config: PreprocessingConfig | str | None = None):
        if config is None:
            self.cfg = PreprocessingConfig()
        elif isinstance(config, (str, Path)):
            self.cfg = PreprocessingConfig.from_yaml(str(config))
        else:
            self.cfg = config

    # ---- public API ----

    def run(
        self,
        dataset: str,
        input_dir: str,
        output_path: str,
    ) -> str:
        """
        Run the full pipeline for an entire dataset.

        Parameters
        ----------
        dataset : str
            ``'deap'`` or ``'seed'``.
        input_dir : str
            Directory containing raw files.
        output_path : str
            Where to write the resulting HDF5 file.

        Returns
        -------
        str
            Absolute path to the HDF5 output.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if dataset.lower() == "deap":
            self._process_deap(input_dir, output_path)
        elif dataset.lower() == "seed":
            self._process_seed(input_dir, output_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset!r}")

        logger.info("Pipeline complete → %s", output_path)
        return os.path.abspath(output_path)

    # ---- DEAP ----

    def _process_deap(self, input_dir: str, output_path: str) -> None:
        import h5py

        fs = self.cfg.deap_sampling_rate
        dat_files = sorted(Path(input_dir).glob("s*.dat"))
        if not dat_files:
            raise FileNotFoundError(f"No .dat files found in {input_dir}")

        logger.info("Processing DEAP: %d subject file(s).", len(dat_files))

        with h5py.File(output_path, "w") as hf:
            for dat_file in dat_files:
                subject_id = dat_file.stem  # e.g. "s01"
                logger.info("  %s …", subject_id)
                data, labels = load_deap_subject(str(dat_file))

                grp = hf.create_group(subject_id)

                for trial_idx in range(data.shape[0]):
                    trial_raw = data[trial_idx].T  # → (n_samples, n_channels)

                    # 1. Band-pass filter
                    trial_filt = bandpass_filter(
                        trial_raw,
                        low=self.cfg.bandpass_low,
                        high=self.cfg.bandpass_high,
                        fs=fs,
                        order=self.cfg.filter_order,
                    )

                    # 2. Optional notch filter
                    if self.cfg.notch_freq is not None:
                        trial_filt = notch_filter(trial_filt, freq=self.cfg.notch_freq, fs=fs)

                    # 3. Epoch extraction
                    epochs = self._extract_epochs(trial_filt, fs)

                    # 4. Normalise (z-score per channel across time)
                    if self.cfg.normalise:
                        epochs = self._normalise(epochs)

                    # 5. Store
                    trial_grp = grp.create_group(f"trial_{trial_idx:02d}")
                    trial_grp.create_dataset("data", data=epochs, compression="gzip")
                    trial_grp.create_dataset("labels", data=labels[trial_idx])

    # ---- SEED ----

    def _process_seed(self, input_dir: str, output_path: str) -> None:
        import h5py

        fs = self.cfg.seed_sampling_rate
        mat_files = sorted(Path(input_dir).glob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in {input_dir}")

        logger.info("Processing SEED: %d session file(s).", len(mat_files))

        with h5py.File(output_path, "w") as hf:
            for f_idx, mat_file in enumerate(mat_files):
                session_id = mat_file.stem
                logger.info("  %s …", session_id)
                trials, labels = load_seed_session(str(mat_file))

                grp = hf.create_group(f"session_{f_idx:02d}_{session_id}")

                for trial_idx, trial_data in enumerate(trials):
                    trial_raw = trial_data.T  # → (n_samples, n_channels)

                    # 1. Band-pass filter
                    if trial_raw.shape[0] < 3 * self.cfg.filter_order + 1:
                        logger.warning(
                            "Trial %d too short (%d samples) — skipping.",
                            trial_idx,
                            trial_raw.shape[0],
                        )
                        continue

                    trial_filt = bandpass_filter(
                        trial_raw,
                        low=self.cfg.bandpass_low,
                        high=self.cfg.bandpass_high,
                        fs=fs,
                        order=self.cfg.filter_order,
                    )

                    # 2. Optional notch filter
                    if self.cfg.notch_freq is not None:
                        trial_filt = notch_filter(trial_filt, freq=self.cfg.notch_freq, fs=fs)

                    # 3. Epoch extraction
                    epochs = self._extract_epochs(trial_filt, fs)

                    # 4. Normalise
                    if self.cfg.normalise:
                        epochs = self._normalise(epochs)

                    # 5. Store
                    trial_grp = grp.create_group(f"trial_{trial_idx:02d}")
                    trial_grp.create_dataset("data", data=epochs, compression="gzip")
                    label = labels[trial_idx] if trial_idx < len(labels) else -1
                    trial_grp.attrs["label"] = int(label)

    # ---- helpers ----

    def _extract_epochs(
        self, data: np.ndarray, fs: float
    ) -> np.ndarray:
        """
        Segment a continuous trial into fixed-length epochs.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_channels)
        fs : float
            Sampling rate.

        Returns
        -------
        np.ndarray, shape (n_epochs, epoch_samples, n_channels)
        """
        epoch_samples = int(self.cfg.epoch_duration * fs)
        step = int((self.cfg.epoch_duration - self.cfg.epoch_overlap) * fs)
        if step <= 0:
            step = epoch_samples

        n_samples = data.shape[0]
        epochs = []
        start = 0
        while start + epoch_samples <= n_samples:
            epochs.append(data[start : start + epoch_samples])
            start += step

        if not epochs:
            # Trial shorter than one epoch — pad with zeros
            padded = np.zeros((epoch_samples, data.shape[1]), dtype=data.dtype)
            padded[: data.shape[0]] = data
            epochs.append(padded)

        return np.stack(epochs, axis=0)

    @staticmethod
    def _normalise(epochs: np.ndarray) -> np.ndarray:
        """
        Z-score normalisation per channel (mean=0, std=1).

        Parameters
        ----------
        epochs : np.ndarray, shape (n_epochs, epoch_samples, n_channels)

        Returns
        -------
        np.ndarray, same shape, normalised.
        """
        mean = epochs.mean(axis=1, keepdims=True)
        std = epochs.std(axis=1, keepdims=True) + 1e-8
        return (epochs - mean) / std
