"""NTIRE 2026 Robust AI-Generated Image Detection training set (Gushchin et al.), extra training data.

Hugging Face `deepfakesMSU/NTIRE-RobustAIGenDetection-train`: shard_0 ... shard_5.zip (~20.6 GB each, shard 5
11.4 GB), each with images/*.jpg and labels.csv (image_name, label; 1 = fake). No generator or distortion labels.
Shard 5 was downloaded first (manifest.csv); this module adds the others, one manifest per shard, so that the
feature cache of every shard stays valid when more shards are added:

  DATASETS_DIR/ntire2026-aigi-train/shard_<k>/images/*.jpg
  DATASETS_DIR/ntire2026-aigi-train/manifest_shard<k>.csv   (path, label, source)

The zip is deleted after a verified extraction (the disk holds about two shards at a time).
Usage: uv run python -m dorian_gray.data.ntire --shards 0 [1 ...] [--keep-zip]
"""

import argparse
import zipfile

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from dorian_gray.config import DATASETS_DIR

REPO = "deepfakesMSU/NTIRE-RobustAIGenDetection-train"
ROOT = DATASETS_DIR / "ntire2026-aigi-train"


def extract(zip_path, shard: int) -> None:
    with zipfile.ZipFile(zip_path) as z:
        for m in tqdm(z.infolist(), desc=f"shard_{shard}"):
            z.extract(m, ROOT)


def manifest(shard: int) -> pd.DataFrame:
    """Manifest from the shard's labels.csv; fails (and keeps the zip) if a labelled image is missing."""
    folder = ROOT / f"shard_{shard}"
    labels = pd.read_csv(folder / "labels.csv", index_col=0)
    images = {p.name for p in (folder / "images").iterdir()}
    missing = set(labels.image_name) - images
    assert not missing, f"shard_{shard}: {len(missing)} labelled images missing"
    df = pd.DataFrame({"path": f"shard_{shard}/images/" + labels.image_name, "label": labels.label.astype(int),
                       "source": f"shard{shard}"})
    df.to_csv(ROOT / f"manifest_shard{shard}.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", required=True)
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    for k in args.shards:
        zip_path = ROOT / f"shard_{k}.zip"
        if not (ROOT / f"shard_{k}" / "labels.csv").exists():
            zip_path = hf_hub_download(REPO, f"shard_{k}.zip", repo_type="dataset", local_dir=ROOT)
            extract(zip_path, k)
        df = manifest(k)
        print(f"shard_{k}: {len(df)} images, {df.label.mean():.3f} fake")
        if not args.keep_zip:
            (ROOT / f"shard_{k}.zip").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
