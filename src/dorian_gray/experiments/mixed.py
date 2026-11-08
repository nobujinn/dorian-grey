"""Train on Community Forensics plus extra labelled data, to test whether more varied training data fixes
the failure to transfer to independent test sets (results log section 10).

Why: models trained on Community Forensics alone learned what *its* real photos look like (original camera
files) and flag re-encoded web photos as fake. Adding a training source whose reals are web images and whose
images carry ordinary distortions (the NTIRE 2026 training set) should teach the real-image variety.

Data:
  * Community Forensics, split per seed exactly as in `classify` (unseen generators held out);
  * each `--extra` set (manifest with path,label[,source]): random 80 / 10 / 10 % train / val / test per seed;
  * each `--test` set: never used for training or thresholds (independent test).
The `val_fpr5` threshold flags 5 % of the pooled validation reals (Community Forensics + extra). Features are
cached per set in cache/features/external/<name>/ (shared with `external`).

Usage: uv run python -m dorian_gray.experiments.mixed --methods a_gbt b [--train-aug] [--input jpeg75] [--no-cf] [--seeds 0 1 2]
       --extra ntire2026_train D:/datasets/ntire2026-aigi-train manifest.csv
       --test ntire2026_val D:/datasets/ntire2026-aigi-val manifest.csv
       --test aigi_now_pixel D:/datasets/aigi-now manifest_pixel.csv
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dorian_gray.config import CACHE_DIR, RUNS_DIR
from dorian_gray.experiments.classify import (OPERATING_POINTS, TARGET_FPR, evaluate, features_matrix,
                                              load_augmented, load_split)
from dorian_gray.experiments.external import extract, load_manifest
from dorian_gray.methods import METHODS
from dorian_gray.metrics import binary_metrics, threshold_at_fpr
from dorian_gray.perturb import AUG_SCHEMES, aug_suffix


def load_set(name: str, root: str, manifest: str, features: str, suffix: str = "") -> pd.DataFrame:
    """Manifest of an external set with its cached features (extracted beforehand by `prefetch`)."""
    df = load_manifest(Path(root), manifest)
    z = np.load(CACHE_DIR / "features" / "external" / (name + suffix) / f"{features}.npz")
    assert list(z["path"]) == list(df.path), f"{name}: cached features out of sync with the manifest"
    df["features"], df["format"] = list(z["features"].astype(np.float32)), z["format"]
    if "source" not in df:
        df["source"] = name
    return df


def prefetch(sets: list[tuple[str, str, str]], features: list[str], normalize: str | None = None,
             augment: str | None = None) -> None:
    """Extract every missing feature cache in one pass over each set's images (augmented copies: `augment`)."""
    suffix = (f"_norm-{normalize}" if normalize else "") + (aug_suffix(augment) if augment else "")
    for name, root, manifest in sets:
        extract(load_manifest(Path(root), manifest), features,
                CACHE_DIR / "features" / "external" / (name + suffix), normalize, augment=augment)


def split_extra(df: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    part = np.full(len(df), "train", dtype=object)
    part[idx[int(0.8 * len(df)):int(0.9 * len(df))]] = "val"
    part[idx[int(0.9 * len(df)):]] = "test"
    return pd.Series(part, index=df.index)


def set_metrics(df: pd.DataFrame, thresholds: dict, dataset: str) -> list[dict]:
    """Metrics on the whole set and per `source`, at every operating point."""
    groups = [("all", np.ones(len(df), bool))] + [(f"source={s}", (df.source == s).to_numpy())
                                                  for s in sorted(df.source.unique()) if df.source.nunique() > 1]
    return [{"threshold": op, "dataset": dataset, "subset": g, **binary_metrics(df.label[m], df.p_fake[m], t)}
            for op, t in thresholds.items() for g, m in groups if df.label[m].nunique() == 2]


def run(args, method_key: str, seed: int, out: Path) -> list[dict]:
    method = METHODS[method_key]
    suffix = f"_norm-{args.input}" if args.input else ""
    extras = [load_set(*e, method.features, suffix).assign(part=lambda d: split_extra(d, seed)) for e in args.extra]
    train = [e[e.part == "train"][["features", "label"]] for e in extras]
    if args.aug_extra:  # augmented copies of the extra sets' training images (same manifest order)
        for e_args, e in zip(args.extra, extras):
            aug = load_set(*e_args, method.features, suffix + aug_suffix(args.aug_scheme))
            train.append(aug[(e.part == "train").to_numpy()][["features", "label"]])
    val_reals = [e[(e.part == "val") & (e.label == 0)] for e in extras]
    cf = None
    if not args.no_cf:
        cf = load_split(method.features + suffix, args.holdout, seed)
        train.append(cf[cf.part == "train"][["features", "label"]])
        if args.train_aug:
            aug = load_augmented(method.features + suffix, cf, args.aug_scheme)
            train.append(aug[aug.part == "train"][["features", "label"]])
    train = pd.concat(train, ignore_index=True)
    model = method.classifier(args.C).fit(features_matrix(train), train["label"])

    def score(df):
        df["p_fake"] = model.predict_proba(features_matrix(df))[:, 1]
        return df

    extras = [score(e) for e in extras]
    pooled_val = [e.loc[(e.part == "val") & (e.label == 0), "p_fake"] for e in extras]
    if cf is not None:
        cf = score(cf)
        pooled_val.append(cf.loc[(cf.part == "val") & (cf.label == 0), "p_fake"])
    thresholds = dict(zip(OPERATING_POINTS, [0.5, threshold_at_fpr(pd.concat(pooled_val), TARGET_FPR)]))

    rows = []
    if cf is not None:
        rows += [{"threshold": op, "dataset": "community_forensics", "subset": k, **v}
                 for op, t in thresholds.items() for k, v in evaluate(cf, t).items()]
    for (name, *_), e in zip(args.extra, extras):
        rows += set_metrics(e[e.part == "val"], thresholds, f"{name}:val")
        rows += set_metrics(e[e.part == "test"], thresholds, f"{name}:test")
    for name, root, manifest in args.test:
        rows += set_metrics(score(load_set(name, root, manifest, method.features, suffix)), thresholds, name)

    out.mkdir(parents=True)
    joblib.dump(model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps({"args": vars(args), "method": method_key, "seed": seed,
                                                  "n_train": len(train), "val_threshold": thresholds["val_fpr5"]},
                                                 indent=2))
    rows = [{"method": method_key, "seed": seed, **r} for r in rows]
    for r in rows:
        if r["subset"] in ("all", "test_unseen") and r["threshold"] == "val_fpr5":
            print(f"{method_key} seed {seed} {r['dataset']:24s} {r['subset']:12s} "
                  f"FPR={r['fpr']:.3f} FNR={r['fnr']:.3f} AUC={r['auc']:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=["a_gbt", "b"])
    ap.add_argument("--extra", nargs=3, action="append", default=[], metavar=("NAME", "ROOT", "MANIFEST"),
                    help="extra labelled set, split 80/10/10 into train/val/test")
    ap.add_argument("--test", nargs=3, action="append", default=[], metavar=("NAME", "ROOT", "MANIFEST"),
                    help="independent test set, never used for training or thresholds")
    ap.add_argument("--no-cf", action="store_true", help="train on the extra sets only (comparison)")
    ap.add_argument("--train-aug", action="store_true", help="also the augmented Community Forensics copies")
    ap.add_argument("--aug-extra", action="store_true", help="also augmented copies of the extra sets' training images")
    ap.add_argument("--aug-scheme", choices=AUG_SCHEMES, default="rj", help="rj: resize + JPEG; deg: + blur, noise")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--input", default=None, help="input normalisation of every image, e.g. jpeg75 (section 9)")
    args = ap.parse_args()
    assert args.extra or not args.no_cf, "nothing to train on"

    features = sorted({METHODS[m].features for m in args.methods})
    prefetch(args.extra + args.test, features, args.input)
    if args.aug_extra:
        prefetch(args.extra, features, args.input, augment=args.aug_scheme)
    tag = ("mixed_" + "+".join(e[0] for e in args.extra) + (aug_suffix(args.aug_scheme) if args.train_aug else "")
           + ("_augextra" if args.aug_extra else "")
           + ("_nocf" if args.no_cf else "") + (f"_norm-{args.input}" if args.input else ""))
    out = RUNS_DIR / tag / datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = [r for m in args.methods for s in args.seeds for r in run(args, m, s, out / m / f"seed{s}")]
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary_per_seed.csv", index=False)
    agg = (summary.groupby(["method", "threshold", "dataset", "subset"])[["accuracy", "fpr", "fnr", "auc"]]
           .agg(["mean", "std"]))
    agg.to_csv(out / "summary.csv")
    main_rows = agg.reset_index()
    main_rows = main_rows[main_rows.subset.isin(["all", "test_unseen"]) & (main_rows.threshold == "val_fpr5")]
    print(main_rows.round(3).to_string(index=False))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
