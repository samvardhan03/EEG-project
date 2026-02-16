"""
Unit Tests for EEG Preprocessing Pipeline
============================================
Tests the pipeline on synthetic data (no real datasets needed).
"""

import os
import tempfile

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="h5py required for pipeline tests")

from src.preprocessing.pipeline import EEGPreprocessor, PreprocessingConfig


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig dataclass."""

    def test_defaults(self):
        cfg = PreprocessingConfig()
        assert cfg.bandpass_low == 0.5
        assert cfg.bandpass_high == 50.0
        assert cfg.epoch_duration == 1.0
        assert cfg.normalise is True

    def test_from_yaml(self, tmp_path):
        yaml_content = """
bandpass_low: 1.0
bandpass_high: 45.0
epoch_duration: 2.0
normalise: false
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        cfg = PreprocessingConfig.from_yaml(str(yaml_file))
        assert cfg.bandpass_low == 1.0
        assert cfg.bandpass_high == 45.0
        assert cfg.epoch_duration == 2.0
        assert cfg.normalise is False


class TestEpochExtraction:
    """Tests for the epoch extraction helper."""

    def test_non_overlapping_epochs(self):
        cfg = PreprocessingConfig(epoch_duration=1.0, epoch_overlap=0.0)
        pp = EEGPreprocessor(config=cfg)

        # 10 seconds at 128 Hz, 4 channels
        data = np.random.randn(1280, 4)
        epochs = pp._extract_epochs(data, fs=128.0)

        assert epochs.shape == (10, 128, 4)  # 10 epochs of 128 samples

    def test_overlapping_epochs(self):
        cfg = PreprocessingConfig(epoch_duration=1.0, epoch_overlap=0.5)
        pp = EEGPreprocessor(config=cfg)

        data = np.random.randn(1280, 4)
        epochs = pp._extract_epochs(data, fs=128.0)

        # With 50% overlap: step = 64 samples → (1280 - 128) / 64 + 1 = 19 epochs
        assert epochs.shape[0] == 19
        assert epochs.shape[1] == 128
        assert epochs.shape[2] == 4

    def test_short_signal_padding(self):
        """Signals shorter than one epoch should be zero-padded."""
        cfg = PreprocessingConfig(epoch_duration=1.0)
        pp = EEGPreprocessor(config=cfg)

        data = np.random.randn(50, 4)  # Shorter than 128 samples
        epochs = pp._extract_epochs(data, fs=128.0)

        assert epochs.shape == (1, 128, 4)
        # Last samples should be zero-padded
        assert np.allclose(epochs[0, 50:, :], 0.0)


class TestNormalise:
    """Tests for z-score normalisation."""

    def test_zero_mean_unit_std(self):
        epochs = np.random.randn(5, 128, 4) * 10 + 3  # Non-zero mean, large std

        normed = EEGPreprocessor._normalise(epochs)

        # Mean should be ~0, std should be ~1 (per epoch, per channel)
        means = normed.mean(axis=1)
        stds = normed.std(axis=1)
        assert np.allclose(means, 0.0, atol=1e-6)
        assert np.allclose(stds, 1.0, atol=1e-2)


class TestPipelineIntegration:
    """Integration test: run pipeline on synthetic DEAP-like data."""

    def test_synthetic_deap_pipeline(self, tmp_path):
        """Create fake DEAP .dat files and run the full pipeline."""
        import pickle

        # Create synthetic data matching DEAP format
        input_dir = tmp_path / "raw"
        input_dir.mkdir()

        for subj in range(1, 3):
            fake_data = {
                "data": np.random.randn(40, 40, 8064).astype(np.float32),
                "labels": np.random.uniform(1, 9, size=(40, 4)).astype(np.float32),
            }
            with open(input_dir / f"s{subj:02d}.dat", "wb") as f:
                pickle.dump(fake_data, f)

        # Run pipeline
        output_path = str(tmp_path / "output.h5")
        cfg = PreprocessingConfig(
            bandpass_low=1.0,
            bandpass_high=45.0,
            notch_freq=None,  # Skip notch for speed
            epoch_duration=1.0,
        )
        pp = EEGPreprocessor(config=cfg)
        pp.run("deap", str(input_dir), output_path)

        # Verify HDF5 structure
        assert os.path.exists(output_path)
        with h5py.File(output_path, "r") as hf:
            assert "s01" in hf
            assert "s02" in hf
            assert "trial_00" in hf["s01"]

            trial_data = hf["s01/trial_00/data"][:]
            assert trial_data.ndim == 3  # (n_epochs, epoch_samples, n_channels)
            assert trial_data.shape[1] == 128  # 1s epochs @ 128 Hz
            assert trial_data.shape[2] == 32   # 32 EEG channels

            labels = hf["s01/trial_00/labels"][:]
            assert labels.shape == (4,)  # [valence, arousal, dominance, liking]
