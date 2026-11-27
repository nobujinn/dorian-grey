"""CocoGlide (Guillaro et al., CVPR 2023, TruFor paper): local AI edits of real photos.

512 MS-COCO photos and, for each, a version in which one object region was repainted by the GLIDE diffusion model
(text-guided inpainting, the prompt being the object's COCO class), plus a binary mask of the repainted region.
All images are 256x256 PNG RGB in both classes, so file format and size carry no label information.
Download: the link in the TruFor repository README (github.com/grip-unina/TruFor), non-profit use only; unpack
the zip into <datasets>/cocoglide/raw (folders real/, fake/, mask/ and one CSV with the real,fake,mask,prompt pairs).

`manifest.csv` (path, label, source, mask, edit_fraction): real images have source `real`; each edited image gets
a source by the fraction of its pixels inside the mask (< 10 %, 10-25 %, > 25 %), so `external` reports each size
group against all real images.

Usage: uv run python -m dorian_gray.data.cocoglide
"""

import csv

import numpy as np
from PIL import Image

from dorian_gray.config import DATASETS_DIR

ROOT = DATASETS_DIR / "cocoglide"
BINS = [(0.10, "edit_small_lt10pct"), (0.25, "edit_medium_10-25pct"), (1.01, "edit_large_gt25pct")]


def edit_fraction(mask_path) -> float:
    return float((np.asarray(Image.open(mask_path).convert("L")) > 127).mean())


def size_group(fraction: float) -> str:
    return next(name for upper, name in BINS if fraction < upper)


def build_manifest(root=ROOT) -> int:
    pairs_csv = next((root / "raw").glob("*.csv"))
    with open(pairs_csv, newline="") as f:
        pairs = list(csv.DictReader(f))
    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source", "mask", "edit_fraction"])
        for p in pairs:
            for key in ("real", "fake", "mask"):
                assert (root / "raw" / p[key]).exists(), f"missing {p[key]}"
            fr = edit_fraction(root / "raw" / p["mask"])
            w.writerow([f"raw/{p['real']}", 0, "real", "", ""])
            w.writerow([f"raw/{p['fake']}", 1, size_group(fr), f"raw/{p['mask']}", f"{fr:.4f}"])
    return len(pairs)


if __name__ == "__main__":
    print(f"{build_manifest()} real/edited pairs -> {ROOT / 'manifest.csv'}")
