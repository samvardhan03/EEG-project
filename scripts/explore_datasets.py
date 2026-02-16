#!/usr/bin/env python3
"""
Dataset Exploration Script
===========================
Quick inspection of raw DEAP / SEED dataset files.
Prints shapes, channel info, label distributions, and basic statistics.

Usage
-----
    python scripts/explore_datasets.py --dataset deap --data-dir data/raw/deap
    python scripts/explore_datasets.py --dataset seed --data-dir data/raw/seed
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def explore_deap(data_dir: str, max_subjects: int = 3) -> None:
    """Inspect DEAP preprocessed .dat files."""
    from src.preprocessing.pipeline import load_deap_subject

    dat_files = sorted(Path(data_dir).glob("s*.dat"))
    if not dat_files:
        print(f"[ERROR] No .dat files found in {data_dir}")
        print("  → Download DEAP preprocessed data first. See data/README.md")
        return

    print(f"\n{'='*60}")
    print(f"  DEAP Dataset — {len(dat_files)} subject file(s)")
    print(f"{'='*60}\n")

    for dat in dat_files[:max_subjects]:
        print(f"─── {dat.name} ───")
        data, labels = load_deap_subject(str(dat))
        print(f"  Data shape  : {data.shape}  (trials × channels × samples)")
        print(f"  Labels shape: {labels.shape}  (trials × [val, aro, dom, lik])")
        print(f"  Data dtype  : {data.dtype}")
        print(f"  Value range : [{data.min():.2f}, {data.max():.2f}]")
        print(f"  Label ranges:")
        for i, name in enumerate(["Valence", "Arousal", "Dominance", "Liking"]):
            vals = labels[:, i]
            print(f"    {name:10s}: mean={vals.mean():.2f}, "
                  f"std={vals.std():.2f}, range=[{vals.min():.1f}, {vals.max():.1f}]")

        # Binarised class distribution
        arousal_high = (labels[:, 1] >= 5).sum()
        valence_pos = (labels[:, 0] >= 5).sum()
        print(f"  Binary distribution (threshold=5):")
        print(f"    Arousal  : High={arousal_high}, Low={40 - arousal_high}")
        print(f"    Valence  : Pos ={valence_pos},  Neg={40 - valence_pos}")
        print()

    if len(dat_files) > max_subjects:
        print(f"  … and {len(dat_files) - max_subjects} more subject(s).\n")


def explore_seed(data_dir: str, max_sessions: int = 3) -> None:
    """Inspect SEED .mat session files."""
    from src.preprocessing.pipeline import load_seed_session

    mat_files = sorted(Path(data_dir).glob("*.mat"))
    if not mat_files:
        print(f"[ERROR] No .mat files found in {data_dir}")
        print("  → Download SEED data first. See data/README.md")
        return

    print(f"\n{'='*60}")
    print(f"  SEED Dataset — {len(mat_files)} session file(s)")
    print(f"{'='*60}\n")

    for mat in mat_files[:max_sessions]:
        print(f"─── {mat.name} ───")
        trials, labels = load_seed_session(str(mat))
        print(f"  Trials      : {len(trials)}")
        if trials:
            shapes = [t.shape for t in trials]
            print(f"  Trial shapes: {shapes[0]} … {shapes[-1]} (channels × samples)")
            total_samples = sum(s[1] for s in shapes)
            print(f"  Total samples: {total_samples:,}")
        print(f"  Labels      : {labels.tolist()}")
        label_counts = Counter(labels.tolist())
        print(f"  Distribution: {dict(label_counts)}  "
              f"(0=neg, 1=neu, 2=pos)")
        print()

    if len(mat_files) > max_sessions:
        print(f"  … and {len(mat_files) - max_sessions} more session(s).\n")


def main():
    parser = argparse.ArgumentParser(
        description="Explore raw EEG datasets (DEAP / SEED).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/explore_datasets.py --dataset deap --data-dir data/raw/deap
  python scripts/explore_datasets.py --dataset seed --data-dir data/raw/seed
  python scripts/explore_datasets.py --dataset deap --data-dir data/raw/deap --max 5
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["deap", "seed"],
        help="Which dataset to explore.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to directory containing raw dataset files.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=3,
        help="Max number of files to inspect (default: 3).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Directory not found: {args.data_dir}")
        sys.exit(1)

    if args.dataset == "deap":
        explore_deap(args.data_dir, max_subjects=args.max)
    else:
        explore_seed(args.data_dir, max_sessions=args.max)


if __name__ == "__main__":
    main()
