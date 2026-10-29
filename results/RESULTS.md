# Results log

Running log of experiments and findings. Every number here comes from a run in this repository;
the tables next to this file are exported from `runs/` with
`uv run python -m dorian_gray.experiments.export`. Label convention: 1 = fake (positive class).
FPR = real images flagged as fake, FNR = fake images passed as real. The decision threshold is
fixed at 0.5 and never tuned on test data.

| Folder | Content |
|---|---|
| `method_b/` | Method B, 5 split seeds: `summary.csv` (mean, std), `summary_per_seed.csv`, per seed `metrics.json`, `per_architecture.csv`, `per_format.csv` |
| `robustness_b/` | Method B under JPEG recompression and resizing: `metrics.csv`, `flagged_by_format.csv` |
| `figures/` | `robustness.pdf` (for LaTeX), `robustness.png` |

---

## 1. Data (29. 10. 2026)

**Community Forensics** (Park & Owens, CVPR 2025), `OwensLab/CommunityForensics-Small`, CC BY-NC-SA.

- The "Small" repository is ~260 GB in 186 parquet shards (not 5.7 GB as the dataset viewer
  suggests). Each shard is one row group whose image column is a single uncompressed page, so a shard
  cannot be read partially: peak RAM is about the shard size (up to ~4.6 GB). We therefore built a
  metadata manifest of all shards (read remotely, metadata columns only) and downloaded 33 selected
  shards (31 GB).
- Seeded sample: 2,000 images per real shard, 1,000 per fake shard → **42,000 images**:

| | JPEG | PNG | Total |
|---|---|---|---|
| Real | 8,000 | 10,000 | 18,000 |
| Fake | 2,077 | 21,923 | 24,000 |

| Fake architecture | Images | Generators |
|---|---|---|
| GAN | 12,126 | 12 |
| Latent diffusion | 7,344 | 311 |
| Pixel diffusion | 2,453 | 3 |
| Other | 2,077 | 1 |

- **Split protocol** (`experiments/classify.py`): real images random 70/10/20 % (train/val/test);
  30 % of the fake *generators* are held out entirely ("unseen" test), the images of the remaining
  generators are split 70/10/20 % ("seen" test). Unseen results pair the held-out fakes with the
  same real test images, so FPR is identical in both test sets.
- The holdout is 30 % of generators *by count* (98 of 327). Since 311 generators are latent-diffusion
  fine-tunes, while GANs are few but large, the number of unseen images varies a lot between seeds
  (4,259 to 10,068), and so does the difficulty of the unseen set. Results are therefore reported as
  mean ± std over 5 seeds.

**Data caveats**
- File names repeat within a shard (different generators reuse names such as `00000152.png`); images
  are keyed by `(shard, row)`. Found by the 1:1 merge check; all 33 caches were verified name by name
  against the manifest.
- All 2,077 JPEG fakes come from **one generator (LFM)**. In the clean test, file format and
  generator are therefore confounded (see 2.3).

## 2. Method B: frozen CLIP ViT-L/14 + logistic regression (UnivFD-style)

Setup: open_clip `ViT-L-14/openai`, visual tower only, fp16 on a GTX 1050 (0.83 GB VRAM, ~7 img/s);
center crop 224 (no downscaling; upscale only below 224 px); 768-d embedding, L2-normalised;
logistic regression, C = 1, balanced class weights. Features cached once per shard.

### 2.1 Seen vs. unseen generators (5 seeds)

| Test set | Accuracy | FPR | FNR | AUC |
|---|---|---|---|---|
| Seen generators | 0.968 ± 0.005 | 0.030 ± 0.005 | 0.033 ± 0.007 | 0.996 ± 0.001 |
| Unseen generators | 0.923 ± 0.033 | 0.030 ± 0.005 | 0.105 ± 0.055 | 0.985 ± 0.008 |

Per seed, unseen FNR = 0.102, 0.028, 0.171, 0.082, 0.140 (seeds 0–4).

**Findings**
- On generators seen in training the probe is near-saturated (AUC 0.996). On unseen generators the
  miss rate triples on average, and its spread across seeds (0.028–0.171) is larger than its mean
  difference to the seen set. A single split would be misleading, so every result uses several seeds.
- FPR does not change between the two test sets (same real images), so generalisation failures show
  up only as missed fakes.

### 2.2 Detection rate of fakes per architecture (unseen generators)

| Architecture | Detected (range over seeds where present) |
|---|---|
| GAN | 0.849 – 0.982 |
| Latent diffusion | 0.783 – 0.960 |
| Pixel diffusion | 0.966 – 0.978 |
| Other (tamingTransformers) | 0.979 |

Unseen **latent-diffusion** generators are the weak spot. Worst single unseen generators (seed 0,
n ≥ 30): LFM 0.684 (JPEG), StyleGAN2-ADA 0.896, GigaGAN 0.917.

### 2.3 "JPEG = real" shortcut, clean test (seed 0)

| | JPEG | PNG |
|---|---|---|
| Real images flagged fake | 0.019 (n = 1,567) | 0.039 (n = 2,033) |
| Fake images detected | 0.684 (n = 2,077, all LFM) | 0.957 (n = 10,778) |

Consistent with the shortcut described by Grommelt et al. (`grommelt2024fake`), but on clean data it
cannot be separated from the generator (all JPEG fakes are LFM). Section 2.4 separates the two.

### 2.4 Robustness to JPEG recompression and resizing (seed 0)

Fixed subset of the test set: 1,000 real, 1,000 fakes of seen generators, 1,000 fakes of unseen
generators. Perturbations applied to real and fake images alike (Pillow; JPEG with 4:2:0 chroma
subsampling; bicubic, antialiased resize); the probe is trained on clean images only.

| Condition | FPR | FNR seen | FNR unseen | AUC seen | AUC unseen |
|---|---|---|---|---|---|
| clean | 0.034 | 0.034 | 0.103 | 0.995 | 0.984 |
| JPEG q95 | 0.017 | 0.176 | 0.268 | 0.977 | 0.967 |
| JPEG q85 | 0.022 | 0.281 | 0.354 | 0.959 | 0.946 |
| JPEG q75 | 0.010 | 0.390 | 0.418 | 0.952 | 0.945 |
| JPEG q50 | 0.015 | 0.462 | 0.585 | 0.933 | 0.905 |
| resize ×0.5 | 0.162 | 0.039 | 0.038 | 0.974 | 0.969 |
| resize ×0.75 | 0.092 | 0.026 | 0.032 | 0.991 | 0.987 |
| resize ×1.5 | 0.192 | 0.015 | 0.042 | 0.987 | 0.971 |

Rate of images flagged as fake, by group and *original* file format:

| Condition | Fake seen, PNG | Fake unseen, JPEG (LFM) | Fake unseen, PNG | Real JPEG | Real PNG |
|---|---|---|---|---|---|
| clean | 0.966 | 0.641 | 0.965 | 0.017 | 0.047 |
| JPEG q95 | 0.824 | 0.593 | 0.769 | 0.026 | 0.010 |
| JPEG q85 | 0.719 | 0.622 | 0.652 | 0.033 | 0.014 |
| JPEG q75 | 0.610 | 0.651 | 0.564 | 0.014 | 0.007 |
| JPEG q50 | 0.538 | 0.306 | 0.444 | 0.024 | 0.009 |
| resize ×0.5 | 0.961 | 1.000 | 0.952 | 0.162 | 0.162 |
| resize ×0.75 | 0.974 | 0.967 | 0.968 | 0.064 | 0.112 |
| resize ×1.5 | 0.985 | 0.856 | 0.985 | 0.124 | 0.241 |

**Findings**
- **JPEG pushes every score towards "real".** Even quality 95 raises the miss rate on seen generators
  5× (0.034 → 0.176); at q50 almost half of the fakes pass (0.462 seen, 0.585 unseen). FPR falls at
  the same time (0.034 → 0.010–0.022).
- **The format shortcut, separated from the generator:** fakes that were PNG lose detection steadily
  under recompression (seen 0.966 → 0.610 at q75), while the LFM fakes, already JPEG, are unaffected
  down to q75 (0.641 → 0.651) and only drop at q50. Real PNG images also become "more real" (flagged
  0.047 → 0.007 at q75). Recompression hurts exactly the images whose format it changes, which
  supports that the probe uses JPEG traces as evidence of authenticity.
- **Resizing has the opposite effect:** it barely affects fakes but makes real images look fake (FPR
  0.162 at ×0.5, 0.192 at ×1.5; 24 % of real PNGs at ×1.5). Interpolation traces are read as
  generator traces.
- **The ranking survives, the threshold does not:** AUC stays ≥ 0.905 under every condition, so most
  of the damage is a shift of the score distribution rather than lost information. This motivates
  (a) training with JPEG/resize augmentation, and (b) calibrating the probability on degraded
  validation images.

![Method B robustness](figures/robustness.png)

## 3. Method A: hand-crafted forensic features + logistic regression

`methods/forensic.py`: 165 named features on a central crop (≤ 256 px, aligned to the 8×8 grid of the
original image, never resampled), in four groups: radial log power spectrum (25), periodic peaks of the
Laplacian residual at k/8 cycle/pixel (12), residual statistics, co-occurrences and inter-channel
correlation (63), 8×8 block-DCT energy and blockiness (65). Standardised logistic regression; the
per-image explanation is coefficient × standardised value per feature. Extraction ~40–46 img/s on CPU.

*Results pending* (feature extraction running 29. 10. 2026): 5-seed classification, feature-group
ablation, robustness, comparison with Method B.

**Design finding:** the SRM `kv3` kernel is separable, (2 − 2 cos ωx)(2 − 2 cos ωy)/4, so its frequency
response is zero on both frequency axes; peaks of purely horizontal or vertical periodic traces would
vanish. The periodic group therefore uses a Laplacian residual (2 cos ωx + 2 cos ωy − 4, non-zero
except at DC). Caught by a unit test with a synthetic period-4 pattern.

## 4. Open issues / next steps

- JPEG fakes from several generators (or JPEG-augmented training) to measure the shortcut on clean data.
- Train-time JPEG/resize augmentation for Method B; calibration (reliability diagram, ECE).
- Tune C on the validation set (currently fixed at 1).
- Method A results, A vs. B comparison on identical images, robustness of Method A.
- Face swap (FF++ face crops) and local manipulation with localisation (CASIA / IMD2020 masks).
