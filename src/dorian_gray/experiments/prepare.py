"""Download the prototype shard selection and cache features of one method for a seeded sample of each shard.

Selection rationale (from the manifest, 28. 10. 2026):
  * real: small JPEG, medium PNG and large PNG shards, so that real-image format and size vary
    and the "JPEG = real" shortcut (`grommelt2024fake`) can be measured via per-format FPR;
  * fake: every architecture in the dataset (latent diffusion, GAN, pixel diffusion, other),
    including mixed shards with many generators, for a held-out-generator test.
Both methods use the same shards and the same seeded rows, so A and B are compared on identical images.

Usage: uv run python -m dorian_gray.experiments.prepare --method a b [--augment] [--normalize jpeg75]
  --augment caches features of one randomly resized / recompressed copy of every image
  (cache name <features>_aug), used as extra training data by `classify --train-aug`.
  --normalize re-encodes every image with one `perturb.CONDITIONS` operation before extraction
  (cache name <features>_norm-<condition>), used by `classify --input <condition>`.
"""

import argparse

import pandas as pd

from dorian_gray.data.community_forensics import LOCAL_DIR, download_shards
from dorian_gray.features import extract_shard
from dorian_gray.methods import METHODS
from dorian_gray.perturb import CONDITIONS

REAL_SHARDS = [
    "HFCF_small_117.parquet", "HFCF_small_125.parquet", "HFCF_small_133.parquet", "HFCF_small_141.parquet",  # JPEG, small
    "HFCF_small_157.parquet", "HFCF_small_165.parquet", "HFCF_small_173.parquet", "HFCF_small_181.parquet",  # PNG, medium
    "HFCF_small_101.parquet",  # PNG, large
]
FAKE_SHARDS = [
    # latent diffusion, ~75 fine-tuned generators each
    "HFCF_small_0.parquet", "HFCF_small_20.parquet", "HFCF_small_40.parquet", "HFCF_small_60.parquet",
    "HFCF_small_78.parquet",  # latent diffusion, JPEG
    # GAN
    "HFCF_small_69.parquet", "HFCF_small_73.parquet", "HFCF_small_74.parquet", "HFCF_small_75.parquet",
    "HFCF_small_80.parquet", "HFCF_small_87.parquet", "HFCF_small_88.parquet",
    # mixed GAN + latent diffusion
    "HFCF_small_66.parquet", "HFCF_small_68.parquet", "HFCF_small_76.parquet", "HFCF_small_77.parquet",
    "HFCF_small_79.parquet",
    # other / GAN + other
    "HFCF_small_70.parquet", "HFCF_small_71.parquet", "HFCF_small_72.parquet",
    # pixel diffusion
    "HFCF_small_67.parquet", "HFCF_small_81.parquet", "HFCF_small_82.parquet", "HFCF_small_83.parquet",
]
MAX_REAL, MAX_FAKE = 2000, 1000
AUGMENT_SEED = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, nargs="+", required=True)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--normalize", choices=[c for c in CONDITIONS if c != "clean"])
    args = ap.parse_args()

    manifest = pd.read_parquet(LOCAL_DIR / "manifest.parquet")
    sizes = manifest.groupby("shard").shard_bytes.first()
    print(f"{len(REAL_SHARDS) + len(FAKE_SHARDS)} shards, {sizes[REAL_SHARDS + FAKE_SHARDS].sum() / 1e9:.1f} GB")
    paths = download_shards(REAL_SHARDS + FAKE_SHARDS)
    suffix = ("_aug" if args.augment else "") + (f"_norm-{args.normalize}" if args.normalize else "")
    extractors = {METHODS[m].features + suffix: (METHODS[m].extractor(), METHODS[m].cache_dtype) for m in args.method}
    for path in paths:
        extract_shard(path, extractors, MAX_REAL if path.name in REAL_SHARDS else MAX_FAKE,
                      augment_seed=AUGMENT_SEED if args.augment else None, normalize=args.normalize)


if __name__ == "__main__":
    main()
