"""Download the prototype shard selection and cache CLIP features for a seeded sample of each shard.

Selection rationale (from the manifest, 28. 10. 2026):
  * real: small JPEG, medium PNG and large PNG shards, so that real-image format and size vary
    and the "JPEG = real" shortcut (`grommelt2024fake`) can be measured via per-format FPR;
  * fake: every architecture in the dataset (latent diffusion, GAN, pixel diffusion, other),
    including mixed shards with many generators, for a held-out-generator test.

Usage: uv run python -m dorian_gray.experiments.prepare_b
"""

import pandas as pd

from dorian_gray.data.community_forensics import LOCAL_DIR, download_shards
from dorian_gray.features import extract_shard
from dorian_gray.methods.clip_probe import ClipEmbedder

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


def main():
    manifest = pd.read_parquet(LOCAL_DIR / "manifest.parquet")
    sizes = manifest.groupby("shard").shard_bytes.first()
    print(f"downloading {len(REAL_SHARDS) + len(FAKE_SHARDS)} shards, "
          f"{sizes[REAL_SHARDS + FAKE_SHARDS].sum() / 1e9:.1f} GB")
    paths = download_shards(REAL_SHARDS + FAKE_SHARDS)
    embedder = ClipEmbedder()
    for path in paths:
        extract_shard(path, embedder, "clip_vitl14", MAX_REAL if path.name in REAL_SHARDS else MAX_FAKE)


if __name__ == "__main__":
    main()
