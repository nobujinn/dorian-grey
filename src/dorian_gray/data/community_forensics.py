"""Community Forensics (Park & Owens, CVPR 2025) on the Hugging Face Hub.

The "Small" repository is ~260 GB in ~100 parquet shards, each a single row group, so we
cannot fetch individual images. Instead we:
  1. build a manifest by reading only the small metadata columns of every shard remotely,
  2. pick a few shards that cover real images and diverse generator architectures,
  3. download just those shards.
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from tqdm import tqdm

from dorian_gray.config import DATASETS_DIR

REPO_ID = "OwensLab/CommunityForensics-Small"
LOCAL_DIR = DATASETS_DIR / "community-forensics-small"
META_COLUMNS = ["image_name", "format", "resolution", "model_name", "label",
                "architecture", "real_source", "subset", "split"]


def build_manifest(out: Path = LOCAL_DIR / "manifest.parquet") -> pd.DataFrame:
    """One row per image (metadata only) plus the shard it lives in and the shard size."""
    fs = HfFileSystem()
    shards = [f for f in fs.ls(f"datasets/{REPO_ID}/data", detail=True) if f["name"].endswith(".parquet")]
    frames = []
    for info in tqdm(shards, desc="shards"):
        with fs.open(info["name"], "rb", block_size=1 << 20) as fh:
            df = pq.ParquetFile(fh).read(columns=META_COLUMNS).to_pandas()
        df["shard"] = Path(info["name"]).name
        df["shard_bytes"] = info["size"]
        frames.append(df)
    manifest = pd.concat(frames, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out)
    return manifest


def load_manifest(path: Path = LOCAL_DIR / "manifest.parquet") -> pd.DataFrame:
    """Manifest plus `row`, the image's position in its shard (manifest rows keep file order)."""
    manifest = pd.read_parquet(path)
    manifest["row"] = manifest.groupby("shard").cumcount()
    return manifest


def download_shards(shards: list[str]) -> list[Path]:
    """Download whole shards (resumable) into LOCAL_DIR/data/."""
    return [Path(hf_hub_download(REPO_ID, f"data/{s}", repo_type="dataset", local_dir=LOCAL_DIR))
            for s in shards]


if __name__ == "__main__":
    m = build_manifest()
    print(m.groupby(["label", "architecture"]).size())
