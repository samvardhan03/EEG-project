"""
Unit Tests for EEG Signal Filters
===================================
Tests band-pass and notch filters on synthetic signals.
"""

import numpy as np
import pytest
from scipy.signal import welch

from src.preprocessing.filters import bandpass_filter, notch_filter


class TestBandpassFilter:
    """Tests for bandpass_filter()."""

    def test_output_shape_matches_input(self):
        """Filtered output must have identical shape."""
        x = np.random.randn(1280, 32)
        y = bandpass_filter(x, low=1.0, high=45.0, fs=128)
        assert y.shape == x.shape

    def test_preserves_in_band_energy(self):
        """A 10 Hz sine (within 1–45 Hz) should survive filtering."""
        fs = 128.0
        t = np.arange(0, 10, 1 / fs)  # 10 seconds
        sine_10hz = np.sin(2 * np.pi * 10 * t)[:, None]  # (1280, 1)

        filtered = bandpass_filter(sine_10hz, low=1.0, high=45.0, fs=fs)

        # Correlation between input and output should be very high
        corr = np.corrcoef(sine_10hz[:, 0], filtered[:, 0])[0, 1]
        assert corr > 0.99, f"In-band signal correlation = {corr:.4f}"

    def test_attenuates_out_of_band(self):
        """A 60 Hz sine (above 50 Hz cutoff) should be heavily attenuated."""
        fs = 128.0
        t = np.arange(0, 10, 1 / fs)
        sine_60hz = np.sin(2 * np.pi * 60 * t)[:, None]

        # Use high=50 so 60 Hz is outside
        filtered = bandpass_filter(sine_60hz, low=1.0, high=50.0, fs=fs)

        # Energy ratio should be small
        energy_in = np.sum(sine_60hz ** 2)
        energy_out = np.sum(filtered ** 2)
        ratio = energy_out / energy_in
        assert ratio < 0.1, f"Out-of-band attenuation ratio = {ratio:.4f}"

    def test_raises_on_invalid_frequencies(self):
        """low >= high should raise ValueError."""
        x = np.random.randn(256, 4)
        with pytest.raises(ValueError, match="low.*must be < high"):
            bandpass_filter(x, low=50.0, high=10.0, fs=128)

    def test_raises_on_nyquist_violation(self):
        """high >= Nyquist should raise ValueError."""
        x = np.random.randn(256, 4)
        with pytest.raises(ValueError, match="Nyquist"):
            bandpass_filter(x, low=1.0, high=65.0, fs=128)

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        x = np.random.randn(512, 8).astype(np.float32)
        y = bandpass_filter(x, low=1.0, high=45.0, fs=128)
        assert y.dtype == np.float32


class TestNotchFilter:
    """Tests for notch_filter()."""

    def test_output_shape(self):
        x = np.random.randn(1280, 32)
        y = notch_filter(x, freq=50.0, fs=128)
        assert y.shape == x.shape

    def test_removes_target_frequency(self):
        """A 50 Hz component should be attenuated by the notch at 50 Hz."""
        fs = 256.0  # Use higher fs to be well above 50 Hz
        t = np.arange(0, 5, 1 / fs)
        # Mix: 10 Hz + 50 Hz noise
        signal = (np.sin(2 * np.pi * 10 * t) +
                  0.5 * np.sin(2 * np.pi * 50 * t))[:, None]

        filtered = notch_filter(signal, freq=50.0, fs=fs)

        # Check PSD: 50 Hz peak should be reduced
        freqs_orig, psd_orig = welch(signal[:, 0], fs=fs, nperseg=512)
        freqs_filt, psd_filt = welch(filtered[:, 0], fs=fs, nperseg=512)

        # Find power at 50 Hz
        idx_50 = np.argmin(np.abs(freqs_orig - 50))
        power_reduction = psd_orig[idx_50] / (psd_filt[idx_50] + 1e-12)
        assert power_reduction > 10, f"50 Hz power reduced by only {power_reduction:.1f}x"
