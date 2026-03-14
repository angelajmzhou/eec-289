"""
Final.py
========
Main entry point / script runner for the non-parametric audio synthesiser.

Imports every public symbol from the project modules so this file (or the
companion notebook) can access everything via a single wildcard import or
by running this script directly.
"""

import os
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from audio_extract import *   # feature extraction, ZCA, K-means, spectral decomp
from Algos import *           # synthesis algorithms, device global, SMT re-exports
from eval_metrics import *    # KL-MFCC, patch diversity, composite score, …
from grid_search import *     # fista_sparse_numpy, run_grid_config


def setup(seed: int = 0) -> None:
    """Seed all RNGs and (re-)configure the shared device."""
    set_seed(seed)         # calls random / numpy / torch seeds (defined in Algos)


if __name__ == "__main__":
    setup()
    setup_audio_dataset(Path("~/Code/eec-289/Final-Proj").expanduser(), "urbansound8k.zip")



