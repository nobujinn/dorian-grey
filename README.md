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

## Experiments

Community Forensics (Park & Owens, CVPR 2025), `OwensLab/CommunityForensics-Small` on the
Hugging Face Hub. Features of each method are extracted once and cached in `cache/`; results go
to `runs/` (both git-ignored).

```
uv run python -m dorian_gray.data.community_forensics            # metadata manifest of all shards
uv run python -m dorian_gray.experiments.prepare --method b      # download shards, cache features
uv run python -m dorian_gray.experiments.classify --method b     # train + evaluate over 5 split seeds
uv run python -m dorian_gray.experiments.robustness --method b   # JPEG, resize, blur, noise, WebP
```

`--method a` runs the same pipeline for Method A (CPU only); `classify --method a --groups spectrum`
trains on one feature group (ablation); `--method a_gbt` uses gradient-boosted trees instead of
logistic regression on the same cached features. Fake generators are held out per split seed, so every
run reports seen-generator and unseen-generator results separately, at the threshold 0.5 and at the
threshold that flags 5 % of the validation reals. `--train-aug` (after `prepare --augment`) also trains on
randomly recompressed / resized copies.

```
uv run python -m dorian_gray.experiments.explain --run method_a_gbt_aug   # Method A explanations per feature group
uv run python -m dorian_gray.experiments.figures                          # thesis figures
```

Independent test sets (never used for training or tuning) are evaluated with the saved seed-0 models
and the thresholds fixed on Community Forensics: a folder with `real/` and `fake/` subfolders, or a CSV
manifest with `path,label[,source]`.

```
uv run python -m dorian_gray.experiments.external --name <set> --root D:/datasets/<set>
```

```
uv run pytest
```

Findings and result tables: [results/RESULTS.md](results/RESULTS.md)
(refresh the tables with `uv run python -m dorian_gray.experiments.export`).
