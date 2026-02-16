"""Preprocessing sub-package: filtering, artifact removal, feature extraction."""
from .filters import bandpass_filter, notch_filter

# Lazy imports for modules with heavy dependencies (h5py, mne)
def __getattr__(name):
    if name == "remove_artifacts_ica":
        from .artifacts import remove_artifacts_ica
        return remove_artifacts_ica
    if name == "EEGPreprocessor":
        from .pipeline import EEGPreprocessor
        return EEGPreprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
