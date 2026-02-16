#!/usr/bin/env bash
# ============================================================
# EEG Emotion Recognition — Environment Setup
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
PYTHON="${PYTHON:-python3}"

echo "=========================================="
echo "  EEG Emotion Recognition — Setup"
echo "=========================================="

# ---- 1. Create virtual environment ----
if [ ! -d "${VENV_DIR}" ]; then
    echo "[1/4] Creating virtual environment..."
    ${PYTHON} -m venv "${VENV_DIR}"
else
    echo "[1/4] Virtual environment already exists."
fi

# ---- 2. Activate & upgrade pip ----
echo "[2/4] Activating venv & upgrading pip..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel --quiet

# ---- 3. Install dependencies ----
echo "[3/4] Installing dependencies (this may take a few minutes)..."
pip install -r "${PROJECT_DIR}/requirements.txt" --quiet

# ---- 4. Create project directories ----
echo "[4/4] Ensuring project directories exist..."
mkdir -p "${PROJECT_DIR}/data/raw"
mkdir -p "${PROJECT_DIR}/data/processed"
mkdir -p "${PROJECT_DIR}/config/model"
mkdir -p "${PROJECT_DIR}/config/training"
mkdir -p "${PROJECT_DIR}/notebooks"
mkdir -p "${PROJECT_DIR}/scripts"
mkdir -p "${PROJECT_DIR}/results"
mkdir -p "${PROJECT_DIR}/checkpoints"
mkdir -p "${PROJECT_DIR}/tests"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "  Activate the environment with:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "  Quick smoke test:"
echo "    python -c \"import mne; import torch; print('All OK')\""
echo ""
