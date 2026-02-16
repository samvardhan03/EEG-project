"""
EEG-Based Emotion Recognition — Package Setup
"""
from setuptools import setup, find_packages

setup(
    name="eeg-emotion-recognition",
    version="0.1.0",
    description="EEG-Based Emotion Recognition with Fourier Adjacency Transformer",
    author="Shekhawat",
    python_requires=">=3.9",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "mne>=1.6",
        "scipy>=1.11",
        "PyWavelets>=1.4",
        "torch>=2.1",
        "scikit-learn>=1.3",
        "numpy>=1.24",
        "pandas>=2.1",
        "h5py>=3.9",
        "matplotlib>=3.8",
        "pyyaml>=6.0",
        "tqdm>=4.66",
    ],
)
