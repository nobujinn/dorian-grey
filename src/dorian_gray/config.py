"""Project-wide paths and device selection.

Datasets live outside the repository. Override the default location with the
DORIAN_DATASETS environment variable (e.g. on the second laptop or on Kaggle).
"""

import os
from pathlib import Path

import torch

DATASETS_DIR = Path(os.environ.get("DORIAN_DATASETS", "D:/datasets"))

# Cached features and experiment outputs: inside the repo folder, but git-ignored.
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "cache"
RUNS_DIR = REPO_ROOT / "runs"


def device() -> torch.device:
    """CUDA if available (GTX 1050 laptop), otherwise CPU (second laptop)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
