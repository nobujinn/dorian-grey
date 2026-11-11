"""DF40 (Yan et al., NeurIPS 2024 D&B): face forgeries by 40 methods, as preprocessed face crops (fakes only).

Access by request (CC BY-NC 4.0); the zips are downloaded by hand into DATASETS_DIR/df40/:
  test/<method>.zip                                 fake frames, <method>/frames/<pair>/<frame>.png
  real/FaceForensics++_real_data_for_DF40.zip       FF++ c23 real frames, same preprocessing as the fakes
  dataset_json/<method>_ff.json                     DF40's own frame lists per split (train / val / test)

Only the official **test** split of the FF++-identity part (`_ff`) is used: it follows the FF++ video split, so
no identity of our FF++-trained face models (trained on the FF++ train split) appears in it. Real frames are
DF40's own crops, never our YuNet crops, so that real and fake share one preprocessing pipeline.
Every listed test frame must be present in the zip (the zips behind DF40's links do not all contain the test
split; such methods are rejected). Output:
  DATASETS_DIR/df40/faces/<method>/<pair>/<frame>.png, faces/real/<video>/<frame>.png
  DATASETS_DIR/df40/manifest.csv   (path, label, source, video)

Usage: uv run python -m dorian_gray.data.df40 --methods simswap inswap
"""

import argparse
import json
import zipfile

import pandas as pd
from tqdm import tqdm

from dorian_gray.config import DATASETS_DIR

ROOT = DATASETS_DIR / "df40"
REAL_ZIP = ROOT / "real" / "FaceForensics++_real_data_for_DF40.zip"


def test_frames(method: str, kind: str) -> dict[str, list[str]]:
    """video -> frame paths (as listed by DF40) of the `_ff` test split; kind is "Real" or "Fake"."""
    j = json.loads((ROOT / "dataset_json" / f"{method}_ff.json").read_text())[f"{method}_ff"]
    return {v: d["frames"] for v, d in j[f"{method}_{kind}"]["test"].items()}


def tail(path: str) -> str:
    """<video>/<frame>.png: the part of a frame path shared by DF40's lists and the zips."""
    return "/".join(path.split("/")[-2:])


def extract(zip_path, frames: dict[str, list[str]], out_name: str, label: int, source: str) -> list[dict]:
    with zipfile.ZipFile(zip_path) as z:
        members = {tail(n): n for n in z.namelist() if not n.endswith("/")}
        wanted = [tail(f) for fs in frames.values() for f in fs]
        missing = [w for w in wanted if w not in members]
        assert not missing, f"{zip_path.name}: {len(missing)} of {len(wanted)} test frames missing (not the test split)"
        rows = []
        for w in tqdm(wanted, desc=out_name):
            out = ROOT / "faces" / out_name / w
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(members[w]))
            rows.append({"path": f"faces/{out_name}/{w}", "label": label, "source": source,
                         "video": f"{source}/{w.split('/')[0]}"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    args = ap.parse_args()
    # real test frames: the same FF++ test videos for every method, so take the union of their lists
    real = {}
    for m in args.methods:
        real.update(test_frames(m, "Real"))
    rows = extract(REAL_ZIP, real, "real", 0, "real")
    for m in args.methods:
        rows += extract(ROOT / "test" / f"{m}.zip", test_frames(m, "Fake"), m, 1, m)
    df = pd.DataFrame(rows).drop_duplicates("path")
    df.to_csv(ROOT / "manifest.csv", index=False)
    print(df.groupby("source").agg(frames=("path", "size"), videos=("video", "nunique")).to_string())


if __name__ == "__main__":
    main()
