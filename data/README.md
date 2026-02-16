# Dataset Acquisition Guide

This directory holds raw and processed EEG datasets.

> **Important:** Both DEAP and SEED require access requests.
> Allow 1–3 business days for approval.

---

## 1. DEAP Dataset

| Item | Detail |
|------|--------|
| **Full name** | Database for Emotion Analysis using Physiological signals |
| **Subjects** | 32 participants |
| **Channels** | 32 EEG + 8 peripheral (we use EEG only) |
| **Sampling rate** | 512 Hz (raw) / **128 Hz** (preprocessed) |
| **Trials** | 40 one-minute music-video clips per subject |
| **Labels** | Valence, Arousal, Dominance, Liking (continuous 1-9) |

### Download Steps

1. Go to <https://www.eecs.qmul.ac.uk/mmv/datasets/deap/>
2. Fill in the EULA / access request form
3. Once approved, download the **"Preprocessed data in Python format"** archive
4. Extract into `data/raw/deap/` so you have:
   ```
   data/raw/deap/
   ├── s01.dat
   ├── s02.dat
   ├── ...
   └── s32.dat
   ```

Each `.dat` file is a Python pickle containing:
- `data`: numpy array of shape `(40, 40, 8064)` — 40 trials × (32 EEG + 8 periph) × 8064 samples
- `labels`: numpy array of shape `(40, 4)` — [valence, arousal, dominance, liking]

### Binarisation

For classification we threshold at the midpoint (5.0):
- **Arousal:** ≥ 5 → High (1), < 5 → Low (0)
- **Valence:** ≥ 5 → Positive (1), < 5 → Negative (0)

---

## 2. SEED Dataset

| Item | Detail |
|------|--------|
| **Full name** | SJTU Emotion EEG Dataset |
| **Subjects** | 15 participants |
| **Channels** | 62 EEG (10-20 system) |
| **Sampling rate** | **200 Hz** (downsampled from 1000 Hz) |
| **Trials** | 15 film clips per session, 3 sessions per subject |
| **Labels** | Negative (−1), Neutral (0), Positive (+1) |

### Download Steps

1. Go to <https://bcmi.sjtu.edu.cn/home/seed/>
2. Request access via the online form
3. Download the **"SEED"** dataset (EEG only)
4. Extract into `data/raw/seed/`:
   ```
   data/raw/seed/
   ├── 1_20131027.mat
   ├── 1_20131030.mat
   ├── ...
   └── 15_20140627.mat
   ```

Each `.mat` file contains 15 trial matrices of shape `(62, n_samples)`.

---

## 3. Running the Preprocessing Pipeline

After downloading, run from the project root:

```bash
# Activate environment
source venv/bin/activate

# DEAP
python -m scripts.explore_datasets --dataset deap --data-dir data/raw/deap

# Preprocess → HDF5
python -c "
from src.preprocessing.pipeline import EEGPreprocessor
pp = EEGPreprocessor()
pp.run('deap', 'data/raw/deap', 'data/processed/deap.h5')
"
```

---

## Directory Layout

```
data/
├── raw/
│   ├── deap/           # Downloaded .dat files
│   └── seed/           # Downloaded .mat files
├── processed/
│   ├── deap.h5         # Preprocessed (generated)
│   └── seed.h5         # Preprocessed (generated)
└── README.md           # ← You are here
```
