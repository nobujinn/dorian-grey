# Dorian Gray

Explainable detection of synthetic and manipulated images (AI-generated images, face swaps,
local edits). Bachelor thesis, FIIT STU Bratislava.

For every image the system returns a calibrated probability of synthetic origin and an
explanation (heatmap, frequency spectrum, localisation map of edited regions).

## Methods

- **Method A — analytical:** noise residual and frequency (FFT/DCT) features with a small,
  interpretable classifier; block-wise analysis localises edited regions.
- **Method B — learned representation:** frozen CLIP ViT-L/14 features with a linear classifier
  (UnivFD-style).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 and all dependencies are installed with:

```
uv sync
uv run dorian-gray
```

PyTorch is installed from the CUDA 12.6 index, which still supports older GPUs (e.g. GTX 10xx).
Without an NVIDIA GPU everything runs on the CPU.

## Data

Datasets are **not** part of this repository (their terms of use forbid redistribution).
They are expected in `D:/datasets`, or in the directory set by the `DORIAN_DATASETS`
environment variable.
