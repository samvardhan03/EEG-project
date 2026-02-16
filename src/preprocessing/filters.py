"""
EEG Signal Filtering Utilities
===============================
Band-pass and notch filters for EEG preprocessing using Butterworth IIR design.

All functions operate on numpy arrays with shape [n_samples, n_channels].
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(
    data: np.ndarray,
    low: float = 0.5,
    high: float = 50.0,
    fs: float = 128.0,
    order: int = 5,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth band-pass filter to multi-channel EEG data.

    The filter retains frequencies in [low, high] Hz and attenuates everything
    outside that range.  A zero-phase design (`filtfilt`) is used to avoid
    introducing group delay.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        Raw EEG time series.  Each column is one electrode channel.
    low : float
        Lower cut-off frequency in Hz (default 0.5 — removes slow drift).
    high : float
        Upper cut-off frequency in Hz (default 50.0 — removes line noise
        and high-frequency muscle artefacts).
    fs : float
        Sampling rate in Hz (DEAP preprocessed = 128, SEED = 200).
    order : int
        Filter order (default 5).  Higher → sharper roll-off but risk of
        ringing on short epochs.

    Returns
    -------
    np.ndarray, same shape as *data*
        Band-pass filtered signal.

    Raises
    ------
    ValueError
        If *low* ≥ *high* or either frequency exceeds the Nyquist limit.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(1280, 32)          # 10s at 128 Hz, 32 channels
    >>> y = bandpass_filter(x, low=1.0, high=45.0, fs=128)
    >>> y.shape
    (1280, 32)
    """
    nyquist = 0.5 * fs
    if low >= high:
        raise ValueError(f"low ({low}) must be < high ({high})")
    if high >= nyquist:
        raise ValueError(
            f"high ({high} Hz) must be < Nyquist ({nyquist} Hz) for fs={fs}"
        )

    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    filtered = filtfilt(b, a, data, axis=0)
    return filtered.astype(data.dtype)


def notch_filter(
    data: np.ndarray,
    freq: float = 50.0,
    fs: float = 128.0,
    quality: float = 30.0,
) -> np.ndarray:
    """
    Remove power-line interference with a notch (band-reject) filter.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        EEG time series.
    freq : float
        Centre frequency to reject (50 Hz in Europe/Asia, 60 Hz in Americas).
    fs : float
        Sampling rate in Hz.
    quality : float
        Quality factor Q = freq / bandwidth.  Higher Q → narrower notch.

    Returns
    -------
    np.ndarray, same shape as *data*
        Notch-filtered signal.
    """
    b, a = iirnotch(freq, quality, fs)
    filtered = filtfilt(b, a, data, axis=0)
    return filtered.astype(data.dtype)


def bandpass_filter_mne(
    raw,
    low: float = 0.5,
    high: float = 50.0,
    method: str = "iir",
):
    """
    In-place band-pass filter on an MNE Raw object.

    Thin wrapper that delegates to ``raw.filter()`` with sensible defaults
    for emotion-recognition pipelines.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object (modified in-place).
    low : float
        Lower cut-off (Hz).
    high : float
        Upper cut-off (Hz).
    method : str
        ``'iir'`` (Butterworth) or ``'fir'`` (windowed sinc).

    Returns
    -------
    mne.io.Raw
        The same object, filtered in-place.
    """
    raw.filter(l_freq=low, h_freq=high, method=method, verbose=False)
    return raw
