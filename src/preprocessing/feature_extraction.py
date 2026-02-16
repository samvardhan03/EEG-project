"""
Feature Extraction for EEG Signals
====================================
Modular functions for spectral and temporal feature extraction:
  - FFT power spectral density per frequency band
  - Discrete Cosine Transform (DCT)
  - Discrete Wavelet Transform (DWT)
  - Differential Entropy (DE)

All functions accept numpy arrays of shape [n_samples, n_channels].
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import welch

# Frequency band boundaries (Hz)
FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


def compute_fft_features(
    eeg_signal: np.ndarray,
    sampling_rate: float = 128.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Extract FFT-based power spectral density for 5 canonical frequency bands.

    For each channel, the PSD is estimated via Welch's method and then
    integrated over each frequency band to yield a single power value.

    Parameters
    ----------
    eeg_signal : np.ndarray, shape (n_samples, n_channels)
        Preprocessed EEG epoch.
    sampling_rate : float
        Sampling frequency in Hz.
    bands : dict or None
        Frequency band definitions.  Defaults to standard 5-band split.

    Returns
    -------
    features : np.ndarray, shape (n_channels, 5)
        Band-power features ordered [delta, theta, alpha, beta, gamma].
    """
    if bands is None:
        bands = FREQ_BANDS

    n_channels = eeg_signal.shape[1]
    n_bands = len(bands)
    features = np.zeros((n_channels, n_bands), dtype=np.float64)

    for ch in range(n_channels):
        freqs, psd = welch(
            eeg_signal[:, ch],
            fs=sampling_rate,
            nperseg=min(256, eeg_signal.shape[0]),
            noverlap=None,
        )

        for b_idx, (band_name, (f_low, f_high)) in enumerate(bands.items()):
            mask = (freqs >= f_low) & (freqs < f_high)
            if mask.any():
                features[ch, b_idx] = np.trapezoid(psd[mask], freqs[mask])

    return features


def compute_dct_features(
    eeg_signal: np.ndarray,
    n_coefficients: int = 10,
) -> np.ndarray:
    """
    Discrete Cosine Transform for energy compaction.

    The DCT provides a compact representation by concentrating signal energy
    in the first few coefficients — useful for dimensionality reduction.

    Parameters
    ----------
    eeg_signal : np.ndarray, shape (n_samples, n_channels)
        EEG epoch.
    n_coefficients : int
        Number of DCT coefficients to retain per channel.

    Returns
    -------
    features : np.ndarray, shape (n_channels, n_coefficients)
        First *n_coefficients* DCT coefficients.
    """
    from scipy.fft import dct

    n_channels = eeg_signal.shape[1]
    features = np.zeros((n_channels, n_coefficients), dtype=np.float64)

    for ch in range(n_channels):
        coeffs = dct(eeg_signal[:, ch], type=2, norm="ortho")
        features[ch, :] = coeffs[:n_coefficients]

    return features


def compute_wavelet_features(
    eeg_signal: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
) -> np.ndarray:
    """
    Discrete Wavelet Transform using Daubechies-4.

    Decomposes the signal into approximation and detail coefficients at
    multiple resolution levels.  Returns the energy of each sub-band.

    Parameters
    ----------
    eeg_signal : np.ndarray, shape (n_samples, n_channels)
        EEG epoch.
    wavelet : str
        Wavelet family (default ``'db4'``).
    level : int or None
        Decomposition level.  ``None`` → automatic based on signal length.

    Returns
    -------
    features : np.ndarray, shape (n_channels, level + 1)
        Sub-band energies: [cA_energy, cD_level, cD_level-1, …, cD_1].
    """
    import pywt

    n_channels = eeg_signal.shape[1]

    if level is None:
        level = pywt.dwt_max_level(eeg_signal.shape[0], pywt.Wavelet(wavelet).dec_len)
        level = min(level, 5)  # Cap at 5 to keep feature size manageable

    features = np.zeros((n_channels, level + 1), dtype=np.float64)

    for ch in range(n_channels):
        coeffs = pywt.wavedec(eeg_signal[:, ch], wavelet, level=level)
        for i, c in enumerate(coeffs):
            features[ch, i] = np.sum(c ** 2) / len(c)  # Normalised energy

    return features


def compute_differential_entropy(
    eeg_signal: np.ndarray,
    sampling_rate: float = 128.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Differential Entropy (DE) features per frequency band.

    DE is defined for a Gaussian random variable X ~ N(μ, σ²) as:
        DE(X) = ½ · ln(2πeσ²)

    For each channel and frequency band, the signal is band-pass filtered
    and its variance is used to estimate DE.  DE features have been shown
    to be among the most discriminative for EEG emotion recognition.

    Parameters
    ----------
    eeg_signal : np.ndarray, shape (n_samples, n_channels)
        Preprocessed EEG epoch.
    sampling_rate : float
        Sampling frequency in Hz.
    bands : dict or None
        Frequency band definitions.

    Returns
    -------
    features : np.ndarray, shape (n_channels, n_bands)
        DE values.  Order: [delta, theta, alpha, beta, gamma].

    References
    ----------
    Shi, Lu-Chen, et al. "Differential entropy feature for EEG-based
    emotion classification." BCI 2013.
    """
    from scipy.signal import butter, filtfilt

    if bands is None:
        bands = FREQ_BANDS

    n_channels = eeg_signal.shape[1]
    n_bands = len(bands)
    features = np.zeros((n_channels, n_bands), dtype=np.float64)

    nyquist = 0.5 * sampling_rate

    for b_idx, (band_name, (f_low, f_high)) in enumerate(bands.items()):
        # Skip if band exceeds Nyquist
        if f_high >= nyquist:
            f_high = nyquist - 1.0
        if f_low >= f_high:
            continue

        # Band-pass to isolate sub-band
        b, a = butter(5, [f_low / nyquist, f_high / nyquist], btype="band")
        band_signal = filtfilt(b, a, eeg_signal, axis=0)

        for ch in range(n_channels):
            variance = np.var(band_signal[:, ch])
            if variance > 0:
                features[ch, b_idx] = 0.5 * np.log(2 * np.pi * np.e * variance)
            else:
                features[ch, b_idx] = 0.0

    return features


def extract_all_features(
    eeg_signal: np.ndarray,
    sampling_rate: float = 128.0,
    dct_coefficients: int = 10,
    wavelet: str = "db4",
) -> Dict[str, np.ndarray]:
    """
    Extract all feature types from a single EEG epoch.

    Parameters
    ----------
    eeg_signal : np.ndarray, shape (n_samples, n_channels)
    sampling_rate : float
    dct_coefficients : int
    wavelet : str

    Returns
    -------
    dict
        Keys: ``'fft'``, ``'dct'``, ``'wavelet'``, ``'de'``.
        Each value is an np.ndarray with shape ``(n_channels, n_features)``.
    """
    return {
        "fft": compute_fft_features(eeg_signal, sampling_rate),
        "dct": compute_dct_features(eeg_signal, dct_coefficients),
        "wavelet": compute_wavelet_features(eeg_signal, wavelet),
        "de": compute_differential_entropy(eeg_signal, sampling_rate),
    }
