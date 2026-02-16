# EEG_Emotion_prediction

EEG emotion detection is a technique that uses electroencephalography (EEG) to measure brain activity and identify different emotional states. EEG is a non-invasive method that measures electrical activity in the brain through electrodes placed on the scalp. The electrical activity of the brain is related to different cognitive and emotional processes, and EEG can be used to measure changes in brain activity that occur in response to different emotions.



> Trying to create a A novel architecture combining Fourier analytic decomposition, graph-neural-network structural priors, and Transformer self-attention for state-of-the-art EEG emotion classification.

---

## Quick Start

```bash
# 1. Clone & set up
cd /path/to/eeg
chmod +x setup.sh && ./setup.sh
source venv/bin/activate

# 2. Download datasets (see data/README.md for instructions)
#    Place DEAP .dat files in data/raw/deap/
#    Place SEED .mat files in data/raw/seed/

# 3. Explore data
python scripts/explore_datasets.py --dataset deap --data-dir data/raw/deap

# 4. Preprocess → HDF5
python -c "
from src.preprocessing.pipeline import EEGPreprocessor
pp = EEGPreprocessor('config/preprocessing.yaml')
pp.run('deap', 'data/raw/deap', 'data/processed/deap.h5')
"

# 5. Run tests
python -m pytest tests/ -v
```

---

## Project Structure

```
eeg/
├── config/                     # YAML configs
│   ├── model/                  # Model architecture configs
│   ├── training/               # Training hyper-parameters
│   └── preprocessing.yaml      # Preprocessing defaults
├── data/
│   ├── raw/                    # Downloaded datasets (gitignored)
│   ├── processed/              # HDF5 outputs
│   └── README.md               # Download guide
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory Data Analysis
├── src/
│   ├── preprocessing/          # Filtering, ICA, pipeline, features
│   ├── models/                 # LSTM, GCN, Transformer, Conformer, FAT
│   ├── graph/                  # Graph construction & GNN layers
│   ├── training/               # Trainer, schedulers
│   ├── evaluation/             # Metrics, visualisation
│   ├── realtime/               # Real-time inference pipeline
│   └── utils/                  # Dataset, helpers, constants
├── scripts/                    # Standalone scripts
├── tests/                      # Unit & integration tests
├── results/                    # Plots, CSVs
├── checkpoints/                # Saved model weights
├── requirements.txt
├── setup.sh
└── setup.py
```

---

## Datasets

| Dataset | Subjects | Channels | Fs (Hz) | Labels |
|---------|----------|----------|---------|--------|
| **DEAP** | 32 | 32 EEG | 128 | Valence / Arousal (binary) |
| **SEED** | 15 | 62 EEG | 200 | Negative / Neutral / Positive |

See [`data/README.md`](data/README.md) for download instructions.

---

