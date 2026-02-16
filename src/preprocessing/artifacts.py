"""
ICA-Based Artifact Removal for EEG
====================================
Automated detection and removal of ocular (eye-blink) and muscular artefacts
using Independent Component Analysis via MNE-Python.

Two strategies are supported:
  1. **Automatic** — correlation-based detection against EOG / frontal channels.
  2. **Semi-automatic** — interactive component inspection (notebook-friendly).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def remove_artifacts_ica(
    raw,
    n_components: int = 15,
    method: str = "fastica",
    eog_channels: Optional[Sequence[str]] = None,
    eog_threshold: float = 0.4,
    random_state: int = 42,
    verbose: bool = False,
):
    """
    Remove eye-blink and muscle artefacts from an MNE Raw object using ICA.

    The function fits ICA, automatically identifies artefactual components
    by correlating each component with designated EOG channels (or frontal
    channels Fp1/Fp2 as a proxy), and subtracts them from the data.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG recording.  **Modified in-place** after artefact
        subtraction.
    n_components : int
        Number of ICA components to estimate.  A value of 15-20 is typical
        for 32-channel montages; increase for 62-channel (SEED).
    method : str
        ICA algorithm: ``'fastica'`` (default), ``'infomax'``, or ``'picard'``.
    eog_channels : list[str] or None
        Channel names to use as EOG reference.  If ``None``, the function
        tries standard names (``'EOG1'``, ``'EOG2'``) then falls back to
        frontal channels (``'Fp1'``, ``'Fp2'``).
    eog_threshold : float
        Pearson-r threshold.  Components with |r| > threshold against any
        EOG channel are marked for exclusion.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        If True, print MNE-level logs.

    Returns
    -------
    raw : mne.io.Raw
        The same Raw object with artefactual components subtracted.
    ica : mne.preprocessing.ICA
        Fitted ICA object (useful for inspection / plotting).

    Notes
    -----
    If no EOG channels are found and no frontal channels exist in the montage,
    the function logs a warning and returns the data unchanged.
    """
    import mne
    from mne.preprocessing import ICA

    mne.set_log_level("WARNING" if not verbose else "INFO")

    # --- Fit ICA ---
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw, verbose=verbose)
    logger.info("ICA fitted with %d components using '%s'.", n_components, method)

    # --- Identify artefactual components ---
    exclude_indices: list[int] = []

    # Try explicit EOG channels first
    candidate_eog = eog_channels or []
    if not candidate_eog:
        # Fall back to standard names present in the data
        for name in ("EOG1", "EOG2", "HEOG", "VEOG", "Fp1", "Fp2"):
            if name in raw.ch_names:
                candidate_eog.append(name)

    if candidate_eog:
        for ch in candidate_eog:
            try:
                indices, scores = ica.find_bads_eog(
                    raw, ch_name=ch, threshold=eog_threshold, verbose=verbose
                )
                exclude_indices.extend(indices)
                logger.info(
                    "EOG channel '%s': %d component(s) flagged (scores: %s).",
                    ch,
                    len(indices),
                    np.round(scores[indices], 3) if len(indices) else "—",
                )
            except Exception as exc:
                logger.warning("EOG detection failed for '%s': %s", ch, exc)
    else:
        logger.warning(
            "No EOG / frontal channels found — skipping automatic artefact "
            "detection.  Consider specifying eog_channels explicitly."
        )

    # De-duplicate
    ica.exclude = sorted(set(exclude_indices))
    logger.info("Excluding %d ICA component(s): %s", len(ica.exclude), ica.exclude)

    # --- Apply (subtract artefacts in-place) ---
    ica.apply(raw, verbose=verbose)

    return raw, ica


def remove_artifacts_autoreject(epochs, random_state: int = 42):
    """
    Drop or interpolate bad epochs using the *autoreject* library.

    This is a complementary strategy to ICA — it works on epoched data and
    can handle artefacts that ICA misses (e.g., electrode pops, brief
    high-amplitude transients).

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data.
    random_state : int
        Seed for the internal cross-validation.

    Returns
    -------
    epochs_clean : mne.Epochs
        Cleaned epochs (some may be dropped entirely).
    reject_log : autoreject.RejectLog
        Log detailing which epochs / channels were repaired or dropped.
    """
    try:
        from autoreject import AutoReject
    except ImportError:
        raise ImportError(
            "autoreject is required for this function.  "
            "Install with: pip install autoreject"
        )

    ar = AutoReject(random_state=random_state, verbose=False)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    logger.info(
        "AutoReject: %d / %d epochs retained.",
        len(epochs_clean),
        len(epochs),
    )
    return epochs_clean, reject_log
