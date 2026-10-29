"""Method B prototype on Community Forensics: frozen CLIP features + logistic-regression probe.

Split protocol (seeded, reproducible):
  * real images: random 70 / 10 / 20 % train / val / test;
  * fake generators (`model_name`): a fraction is held out entirely ("unseen" test set);
    images of the remaining generators are split 70 / 10 / 20 % like the reals ("seen" test set).
So the test results separate in-distribution accuracy from generalisation to unseen generators.
The decision threshold is 0.5 on the probe's probability and is never tuned on test data.

Usage: uv run python -m dorian_gray.experiments.prototype_b [--holdout 0.3] [--seed 0]
"""

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from dorian_gray.config import RUNS_DIR
from dorian_gray.data.community_forensics import LOCAL_DIR, load_manifest
from dorian_gray.features import load_cached
from dorian_gray.methods.clip_probe import make_probe, save_probe
from dorian_gray.metrics import binary_metrics

FEATURES = "clip_vitl14"


def split(df: pd.DataFrame, holdout: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    part = pd.Series("train", index=df.index)

    def random_split(idx):
        idx = rng.permutation(idx)
        n_tr, n_va = int(0.7 * len(idx)), int(0.1 * len(idx))
        part[idx[n_tr:n_tr + n_va]] = "val"
        part[idx[n_tr + n_va:]] = "test_seen"

    random_split(df.index[df.label == 0].to_numpy())
    generators = rng.permutation(sorted(df.loc[df.label == 1, "model_name"].unique()))
    unseen = set(generators[: int(round(holdout * len(generators)))])
    fake = df.label == 1
    part[fake & df.model_name.isin(unseen)] = "test_unseen"
    random_split(df.index[fake & ~df.model_name.isin(unseen)].to_numpy())
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=float, default=0.3, help="fraction of fake generators held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    manifest = load_manifest()
    shards = sorted(p.name for p in (LOCAL_DIR / "data").glob("*.parquet"))
    df = load_cached(FEATURES, shards).merge(manifest, on=["shard", "row"], how="left",
                                                validate="1:1", suffixes=("", "_manifest"))
    assert (df.image_name == df.image_name_manifest).all(), "cached features out of sync with the manifest"
    df["part"] = split(df, args.holdout, args.seed)

    X = np.stack(df["features"].to_numpy()).astype(np.float32)
    tr = (df.part == "train").to_numpy()
    probe = make_probe(args.C).fit(X[tr], df.loc[tr, "label"])
    df["p_fake"] = probe.predict_proba(X)[:, 1]

    real_test = (df.part == "test_seen") & (df.label == 0)
    results = {
        "val": binary_metrics(df.loc[df.part == "val", "label"], df.loc[df.part == "val", "p_fake"]),
        "test_seen": binary_metrics(df.loc[df.part == "test_seen", "label"], df.loc[df.part == "test_seen", "p_fake"]),
        # unseen generators are all fake, so pair them with the real test images
        "test_unseen": binary_metrics(df.loc[real_test | (df.part == "test_unseen"), "label"],
                                      df.loc[real_test | (df.part == "test_unseen"), "p_fake"]),
    }
    test = df.part.isin(["test_seen", "test_unseen"]) & (df.label == 1)
    per_arch = (df[test].groupby(["part", "architecture"])
                .apply(lambda g: pd.Series({"n": len(g), "detected": (g.p_fake >= 0.5).mean()}), include_groups=False)
                .reset_index())
    # "JPEG = real" shortcut check (grommelt2024fake): rate of images flagged fake, per class and file format
    per_format = (df[df.part.isin(["test_seen", "test_unseen"])]
                  .groupby(["label", "format"])
                  .apply(lambda g: pd.Series({"n": len(g), "flagged_fake": (g.p_fake >= 0.5).mean()}), include_groups=False)
                  .reset_index())

    out =RUNS_DIR / "method_b_prototype" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps({"args": vars(args), "shards": shards, **results}, indent=2))
    per_arch.to_csv(out / "per_architecture.csv", index=False)
    per_format.to_csv(out / "per_format.csv", index=False)
    df.drop(columns=["features"]).to_parquet(out / "predictions.parquet")
    save_probe(probe, out / "probe.joblib")

    print(df.groupby(["part", "label"]).size().unstack(fill_value=0))
    for k, v in results.items():
        print(f"{k:12s} acc={v['accuracy']:.3f} FPR={v['fpr']:.3f} FNR={v['fnr']:.3f} AUC={v['auc']:.3f} "
              f"(real={v['n_real']}, fake={v['n_fake']})")
    print(per_arch.to_string(index=False))
    print(per_format.to_string(index=False))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
