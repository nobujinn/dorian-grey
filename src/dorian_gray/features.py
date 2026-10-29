"""Read images from parquet shards and cache backbone features to disk (one .npz per shard)."""

import base64
import io
import zlib
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from dorian_gray.config import CACHE_DIR
from dorian_gray.perturb import random_augment


def decode_image(value) -> Image.Image:
    """Community Forensics stores encoded image files, as raw bytes or as a base64 string."""
    if isinstance(value, str):
        value = base64.b64decode(value.strip('"'))
    return Image.open(io.BytesIO(value))


def sample_rows(n_rows: int, max_images: int | None, seed: int = 0) -> np.ndarray:
    """Seeded subset of row indices, so the same images are picked on every machine."""
    if max_images is None or max_images >= n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, max_images, replace=False))


def iter_shard(path: Path, rows: np.ndarray, batch_size: int = 32) -> Iterator[tuple[pd.DataFrame, list[Image.Image]]]:
    """Yield (names, images) for the selected rows only; unselected images are never decoded.

    Each shard is a single row group whose image column is one uncompressed page, so Arrow
    materialises the whole column (peak RAM ~ shard size, up to ~4.6 GB for the largest shards;
    on the 8 GB laptop Windows pages other programs out). Images are decoded batch by batch.
    """
    pf = pq.ParquetFile(path, memory_map=True)
    wanted = np.zeros(pf.metadata.num_rows, dtype=bool)
    wanted[rows] = True
    names, images, offset = [], [], 0
    for batch in pf.iter_batches(batch_size=256, columns=["image_name", "image_data"]):
        keep = np.flatnonzero(wanted[offset:offset + batch.num_rows])
        offset += batch.num_rows
        if keep.size == 0:
            continue
        sub = batch.take(keep)
        names.extend(sub.column("image_name").to_pylist())
        images.extend(decode_image(v) for v in sub.column("image_data").to_pylist())
        while len(images) >= batch_size:
            yield pd.DataFrame({"image_name": names[:batch_size]}), images[:batch_size]
            names, images = names[batch_size:], images[batch_size:]
    if images:
        yield pd.DataFrame({"image_name": names}), images


def extract_shard(path: Path, extractors: dict[str, tuple], max_images: int | None = None,
                  batch_size: int = 32, augment_seed: int | None = None) -> None:
    """Embed (a seeded sample of) a shard once per extractor; extractors whose cache exists are skipped.

    extractors: cache name -> (callable images -> (N, D) array, storage dtype). Several extractors share
    one read of the shard, and all of them see exactly the same (possibly augmented) images.
    augment_seed: if set, every image is first randomly resized / recompressed (`perturb.random_augment`),
    deterministically per (seed, shard, row); the applied parameters are stored next to the features.
    """
    todo = {n: e for n, e in extractors.items() if not (CACHE_DIR / "features" / n / f"{path.stem}.npz").exists()}
    if not todo:
        return
    rows = sample_rows(pq.ParquetFile(path).metadata.num_rows, max_images)
    shard_key = zlib.crc32(path.stem.encode())
    names, feats, params, pos = [], {n: [] for n in todo}, [], 0
    with tqdm(total=len(rows), desc=path.stem) as bar:
        for meta, images in iter_shard(path, rows, batch_size):
            if augment_seed is not None:
                aug = [random_augment(im, np.random.default_rng([augment_seed, shard_key, int(r)]))
                       for im, r in zip(images, rows[pos:pos + len(images)])]
                images = [im for im, _ in aug]
                params.extend(prm for _, prm in aug)
            pos += len(images)
            for n, (fn, dtype) in todo.items():
                feats[n].append(fn(images).astype(dtype))
            names.extend(meta["image_name"].tolist())
            bar.update(len(images))
    extra = {}
    if params:
        quality, scale = zip(*params)
        extra = {"jpeg_quality": np.array(quality), "scale": np.array(scale)}
    for n in todo:
        out = CACHE_DIR / "features" / n / f"{path.stem}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        # image names repeat within a shard (generators reuse file names), so the row index is the key
        np.savez(out, row=rows, image_name=np.array(names), features=np.concatenate(feats[n]), **extra)


def load_cached(name: str, shards: list[str]) -> pd.DataFrame:
    """Return a frame with columns shard, row, image_name, features (one row per image)."""
    frames = []
    for shard in shards:
        z = np.load(CACHE_DIR / "features" / name / f"{Path(shard).stem}.npz")
        frames.append(pd.DataFrame({"shard": shard, "row": z["row"], "image_name": z["image_name"],
                                    "features": list(z["features"])}))
    return pd.concat(frames, ignore_index=True)
