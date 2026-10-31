"""Evaluate trained models on an independent, external test set (never used for training or tuning).

The models are the seed-0 models of the latest `classify` runs (trained on Community Forensics); the
thresholds are the ones fixed on Community Forensics validation data (0.5 and `val_fpr5`). So the
external set measures transfer to a different data source as it is, without any adaptation.

Input: a folder with `real/` and `fake/` subfolders (searched recursively), or a CSV manifest with
columns `path` (relative to the CSV) and `label` (0 = real, 1 = fake), optionally `source`. Images stay
where they are (outside the repo); features are cached in cache/features/external/<name>/.

Usage: uv run python -m dorian_gray.experiments.external --name chameleon --root D:/datasets/chameleon
       [--manifest labels.csv] [--input jpeg75] [--runs method_a_gbt_aug method_b_aug ...]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from dorian_gray.config import CACHE_DIR, RUNS_DIR
from dorian_gray.methods import METHODS
from dorian_gray.metrics import binary_metrics
from dorian_gray.perturb import apply

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_RUNS = ["method_a", "method_a_gbt_aug", "method_b", "method_b_aug"]


def load_manifest(root: Path, manifest: str | None) -> pd.DataFrame:
    if manifest:
        df = pd.read_csv(root / manifest)
        df["path"] = [str((root / p).resolve()) for p in df["path"]]
    else:
        rows = [{"path": str(p.resolve()), "label": label}
                for label, sub in [(0, "real"), (1, "fake")]
                for p in sorted((root / sub).rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
        df = pd.DataFrame(rows)
    assert len(df) and set(df.label.unique()) <= {0, 1}, "need images labelled 0 (real) / 1 (fake)"
    return df.reset_index(drop=True)


def extract(df: pd.DataFrame, features: list[str], out_dir: Path, normalize: str | None, batch_size: int = 32):
    """One pass over the images for every feature extractor whose cache is missing; also records the format."""
    todo = [f for f in features if not (out_dir / f"{f}.npz").exists()]
    if not todo:
        return
    extractors = {f: next((m.extractor(), m.cache_dtype) for m in METHODS.values() if m.features == f) for f in todo}
    feats, formats = {f: [] for f in todo}, []
    for start in tqdm(range(0, len(df), batch_size), desc="external"):
        images = []
        for p in df.path[start:start + batch_size]:
            im = Image.open(p)
            im.load()
            formats.append(im.format or "unknown")
            images.append(apply(im, normalize) if normalize else im)
        for f, (fn, dtype) in extractors.items():
            feats[f].append(fn(images).astype(dtype))
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in todo:
        np.savez(out_dir / f"{f}.npz", path=df.path.to_numpy(dtype=str), format=np.array(formats),
                 features=np.concatenate(feats[f]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short name of the external set (cache and output folder)")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", default=None, help="CSV with path,label[,source], relative to --root")
    ap.add_argument("--input", default=None, help="input normalisation, must match the runs (e.g. jpeg75)")
    ap.add_argument("--runs", nargs="+", default=None, help="classify experiment folders to evaluate")
    args = ap.parse_args()

    suffix = f"_norm-{args.input}" if args.input else ""
    runs = args.runs or [r + suffix for r in DEFAULT_RUNS]
    df = load_manifest(args.root, args.manifest)
    print(df.groupby("label").size().rename({0: "real", 1: "fake"}).to_string())

    models = {}
    for tag in runs:
        run = sorted(p for p in (RUNS_DIR / tag).iterdir() if p.is_dir())[-1]
        meta = json.loads((run / "seed0" / "metrics.json").read_text())
        assert (meta["args"].get("input") or None) == args.input, f"{tag} was trained with a different input"
        models[tag] = (METHODS[meta["args"]["method"]].features + suffix, joblib.load(run / "seed0" / "model.joblib"),
                       {"0.5": 0.5, "val_fpr5": meta["val_threshold"]}, run.name)

    cache = CACHE_DIR / "features" / "external" / (args.name + suffix)
    extract(df, sorted({f.removesuffix(suffix) for f, *_ in models.values()}), cache, args.input)

    metrics, by_format, preds = [], [], df.copy()
    for tag, (features, model, thresholds, run_name) in models.items():
        z = np.load(cache / f"{features.removesuffix(suffix)}.npz")
        assert list(z["path"]) == list(df.path), "cached features out of sync with the image list"
        p = model.predict_proba(z["features"].astype(np.float32))[:, 1]
        preds[tag], preds["format"] = p, z["format"]
        groups = [("all", np.ones(len(df), bool))] + ([(f"source={s}", (df.source == s).to_numpy())
                                                       for s in sorted(df.source.unique())] if "source" in df else [])
        for op, t in thresholds.items():
            for gname, m in groups:
                if df.label[m].nunique() == 2:
                    metrics.append({"model": tag, "threshold": op, "subset": gname,
                                    **binary_metrics(df.label[m], p[m], t)})
            flagged = pd.DataFrame({"label": df.label, "format": z["format"], "flag": p >= t})
            by_format.append(flagged.groupby(["label", "format"]).flag.agg(["size", "mean"]).reset_index()
                             .assign(model=tag, threshold=op))

    out = RUNS_DIR / f"external_{args.name}{suffix}" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    (out / "args.json").write_text(json.dumps({**vars(args), "root": str(args.root), "n": len(df),
                                               "models": {t: m[3] for t, m in models.items()}}, indent=2))
    metrics, by_format = pd.DataFrame(metrics), pd.concat(by_format)
    metrics.to_csv(out / "metrics.csv", index=False)
    by_format.to_csv(out / "flagged_by_format.csv", index=False)
    preds.drop(columns=["path"]).to_parquet(out / "predictions.parquet")  # no file paths, not exported
    print(metrics[metrics.subset == "all"].drop(columns="subset").round(3).to_string(index=False))
    print(by_format.round(3).to_string(index=False))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
