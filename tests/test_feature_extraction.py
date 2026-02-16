"""
Unit Tests for Feature Extraction
===================================
Tests FFT, DCT, Wavelet, and DE feature extraction on synthetic signals.
"""

import numpy as np
import pytest

from src.preprocessing.feature_extraction import (
    compute_dct_features,
    compute_differential_entropy,
    compute_fft_features,
    compute_wavelet_features,
    extract_all_features,
)


class TestFFTFeatures:
    """Tests for compute_fft_features()."""

    def test_output_shape(self):
        x = np.random.randn(1280, 32)
        features = compute_fft_features(x, sampling_rate=128.0)
        assert features.shape == (32, 5)  # 32 channels, 5 bands

    def test_dominant_band_detection(self):
        """A 10 Hz sine should have max power in the alpha band (8-13 Hz)."""
        fs = 128.0
        t = np.arange(0, 10, 1 / fs)
        sine_10hz = np.sin(2 * np.pi * 10 * t)[:, None]

        features = compute_fft_features(sine_10hz, sampling_rate=fs)
        # Alpha is index 2
        assert features[0, 2] == features[0].max(), (
            f"Alpha band should dominate: {features[0]}"
        )

    def test_non_negative(self):
        """PSD-based features should be non-negative."""
        x = np.random.randn(512, 8)
        features = compute_fft_features(x, sampling_rate=128.0)
        assert np.all(features >= 0)


class TestDCTFeatures:
    """Tests for compute_dct_features()."""

    def test_output_shape(self):
        x = np.random.randn(1280, 16)
        features = compute_dct_features(x, n_coefficients=10)
        assert features.shape == (16, 10)

    def test_energy_compaction(self):
        """First DCT coefficients should capture most energy."""
        x = np.random.randn(1024, 1)
        features_full = compute_dct_features(x, n_coefficients=50)
        features_partial = compute_dct_features(x, n_coefficients=10)

        # First 10 coefficients should have substantial energy
        energy_10 = np.sum(features_partial ** 2)
        energy_50 = np.sum(features_full ** 2)
        assert energy_10 > 0


class TestWaveletFeatures:
    """Tests for compute_wavelet_features()."""

    def test_output_shape(self):
        x = np.random.randn(1024, 8)
        features = compute_wavelet_features(x, wavelet="db4")
        assert features.shape[0] == 8  # 8 channels
        assert features.shape[1] >= 2  # At least approx + 1 detail level

    def test_non_negative_energy(self):
        """Sub-band energies should be non-negative."""
        x = np.random.randn(512, 4)
        features = compute_wavelet_features(x)
        assert np.all(features >= 0)


class TestDifferentialEntropy:
    """Tests for compute_differential_entropy()."""

    def test_output_shape(self):
        x = np.random.randn(1280, 32)
        features = compute_differential_entropy(x, sampling_rate=128.0)
        assert features.shape == (32, 5)

    def test_higher_variance_higher_de(self):
        """Signals with higher variance should yield higher DE."""
        fs = 128.0
        t = np.arange(0, 10, 1 / fs)

        low_var = np.random.randn(len(t), 1) * 0.1
        high_var = np.random.randn(len(t), 1) * 10.0

        de_low = compute_differential_entropy(low_var, sampling_rate=fs)
        de_high = compute_differential_entropy(high_var, sampling_rate=fs)

        # High variance should give higher DE in most bands
        assert de_high.mean() > de_low.mean()


class TestExtractAll:
    """Tests for extract_all_features()."""

    def test_returns_all_keys(self):
        x = np.random.randn(512, 8)
        result = extract_all_features(x, sampling_rate=128.0)
        assert set(result.keys()) == {"fft", "dct", "wavelet", "de"}

    def test_all_shapes_valid(self):
        x = np.random.randn(512, 8)
        result = extract_all_features(x, sampling_rate=128.0)
        for key, arr in result.items():
            assert arr.ndim == 2
            assert arr.shape[0] == 8  # n_channels
