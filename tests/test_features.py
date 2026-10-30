import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from dorian_gray import features


def _shard(path, n=40):
    """Tiny parquet shard; image i is a flat image of gray level i, so the image identifies its row."""
    data = []
    for i in range(n):
        buf = io.BytesIO()
        Image.new("RGB", (48, 32), (i, i, i)).save(buf, format="PNG")
        data.append(buf.getvalue())
    pq.write_table(pa.table({"image_name": [f"{i % 7:08d}.png" for i in range(n)], "image_data": data}), path)


def _mean_gray(images):
    return np.array([[np.asarray(im.convert("L"), float).mean()] for im in images])


def test_extract_shard_rows_features_and_augmentation(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "CACHE_DIR", tmp_path / "cache")
    shard = tmp_path / "HFCF_small_test.parquet"
    _shard(shard)
    extractors = {"plain": (_mean_gray, "float32")}
    features.extract_shard(shard, extractors, max_images=15, batch_size=4)
    z = np.load(tmp_path / "cache/features/plain/HFCF_small_test.npz")
    assert len(z["row"]) == 15 and np.array_equal(z["row"], np.sort(z["row"]))
    assert np.allclose(z["features"][:, 0], z["row"])  # each feature belongs to its own row

    for name in ("aug1", "aug2"):
        features.extract_shard(shard, {name: (_mean_gray, "float32")}, max_images=15, batch_size=4, augment_seed=1)
    a, b = (np.load(tmp_path / f"cache/features/{n}/HFCF_small_test.npz") for n in ("aug1", "aug2"))
    assert np.array_equal(a["jpeg_quality"], b["jpeg_quality"]) and np.array_equal(a["features"], b["features"])
    assert (a["jpeg_quality"] > 0).any() and (a["scale"] != 1).any()
    # augmentation keeps a flat image roughly flat: still identifiable by its gray level
    assert np.abs(a["features"][:, 0] - a["row"]).max() < 2


def test_extract_shard_normalize_reencodes_every_image(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "CACHE_DIR", tmp_path / "cache")
    shard = tmp_path / "HFCF_small_test.parquet"
    _shard(shard)
    seen = []
    def record(images):
        seen.extend(im.format for im in images)
        return _mean_gray(images)
    features.extract_shard(shard, {"norm": (record, "float32")}, max_images=10, batch_size=4, normalize="jpeg75")
    assert seen == ["JPEG"] * 10
