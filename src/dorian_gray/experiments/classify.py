"""Train and evaluate one detection method (A or B) on cached Community Forensics features.

Split protocol (seeded, reproducible):
  * real images: random 70 / 10 / 20 % train / val / test;
  * fake generators (`model_name`): a fraction is held out entirely ("unseen" test set);
    images of the remaining generators are split 70 / 10 / 20 % like the reals ("seen" test set).
So the test results separate in-distribution accuracy from generalisation to unseen generators.
The decision threshold is 0.5 on the probe's probability and is never tuned on test data.

Usage: uv run python -m dorian_gray.experiments.classify --method {a,b} [--seeds 0 1 2 3 4]
       [--groups spectrum periodic residual dct]   (method A only: feature-group ablation)
"""

import argparse
import json
from datetime import datetime

import joblib

import numpy as np
import pandas as pd

from dorian_gray.config import RUNS_DIR
from dorian_gray.data.community_forensics import LOCAL_DIR, load_manifest
from dorian_gray.features import load_cached
from dorian_gray.methods import METHODS
from dorian_gray.metrics import binary_metrics


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


def load_split(features: str, holdout: float, seed: int) -> pd.DataFrame:
    """Cached features of every downloaded shard, joined with the manifest, plus the split column `part`."""
    manifest = load_manifest()
    shards = sorted(p.name for p in (LOCAL_DIR / "data").glob("*.parquet"))
    df = load_cached(features, shards).merge(manifest, on=["shard", "row"], how="left",
                                                validate="1:1", suffixes=("", "_manifest"))
    assert (df.image_name == df.image_name_manifest).all(), "cached features out of sync with the manifest"
    df["part"] = split(df, holdout, seed)
    return df


def features_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["features"].to_numpy()).astype(np.float32)


def fit_probe(df: pd.DataFrame, classifier):
    tr = (df.part == "train").to_numpy()
    return classifier.fit(features_matrix(df[tr]), df.loc[tr, "label"])


def select_groups(df: pd.DataFrame, groups: list[str]) -> list[str]:
    """Method A: keep only the feature columns of the given groups; returns their names."""
    from dorian_gray.methods.forensic import feature_group, feature_names
    names = feature_names()
    cols = [i for i, n in enumerate(names) if feature_group(n) in groups]
    df["features"] = [f[cols] for f in df["features"]]
    return [names[i] for i in cols]


def evaluate(df: pd.DataFrame) -> dict:
    """Metrics on val, seen-generator test and unseen-generator test; needs a `p_fake` column."""
    real_test = (df.part == "test_seen") & (df.label == 0)
    sets = {"val": df.part == "val", "test_seen": df.part == "test_seen",
            # unseen generators are all fake, so pair them with the real test images
            "test_unseen": real_test | (df.part == "test_unseen")}
    return {k: binary_metrics(df.loc[m, "label"], df.loc[m, "p_fake"]) for k, m in sets.items()}


def run(args, seed: int, out) -> dict:
    method = METHODS[args.method]
    df = load_split(method.features, args.holdout, seed)
    names = select_groups(df, args.groups) if args.method == "a" else None
    probe = fit_probe(df, method.classifier(args.C))
    df["p_fake"] = probe.predict_proba(features_matrix(df))[:, 1]
    results = evaluate(df)

    test = df.part.isin(["test_seen", "test_unseen"]) & (df.label == 1)
    per_arch = (df[test].groupby(["part", "architecture"])
                .apply(lambda g: pd.Series({"n": len(g), "detected": (g.p_fake >= 0.5).mean()}), include_groups=False)
                .reset_index())
    # "JPEG = real" shortcut check (grommelt2024fake): rate of images flagged fake, per class and file format
    per_format = (df[df.part.isin(["test_seen", "test_unseen"])]
                  .groupby(["label", "format"])
                  .apply(lambda g: pd.Series({"n": len(g), "flagged_fake": (g.p_fake >= 0.5).mean()}), include_groups=False)
                  .reset_index())

    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps({"args": vars(args), "seed": seed,
                                                  "shards": sorted(df.shard.unique()), **results}, indent=2))
    per_arch.to_csv(out / "per_architecture.csv", index=False)
    per_format.to_csv(out / "per_format.csv", index=False)
    df.drop(columns=["features"]).to_parquet(out / "predictions.parquet")
    joblib.dump(probe, out / "model.joblib")
    if names is not None:  # global explanation: log-odds change per standard deviation of each feature
        coef = pd.DataFrame({"feature": names, "coef": probe[-1].coef_[0]})
        coef.reindex(coef.coef.abs().sort_values(ascending=False).index).to_csv(out / "coefficients.csv", index=False)

    print(f"--- seed {seed}")
    print(df.groupby(["part", "label"]).size().unstack(fill_value=0))
    for k, v in results.items():
        print(f"{k:12s} acc={v['accuracy']:.3f} FPR={v['fpr']:.3f} FNR={v['fnr']:.3f} AUC={v['auc']:.3f} "
              f"(real={v['n_real']}, fake={v['n_fake']})")
    print(per_arch.to_string(index=False))
    print(per_format.to_string(index=False))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--groups", nargs="+", default=["spectrum", "periodic", "residual", "dct"],
                    help="method A feature groups to use (ablation)")
    ap.add_argument("--holdout", type=float, default=0.3, help="fraction of fake generators held out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="one split per seed; which generators are held out varies a lot between seeds")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    tag = f"method_{args.method}" + ("" if args.method == "b" or len(args.groups) == 4 else "_" + "+".join(args.groups))
    out = RUNS_DIR / tag / datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = [{"seed": seed, "set": k, **v} for seed in args.seeds
            for k, v in run(args, seed, out / f"seed{seed}").items()]
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary_per_seed.csv", index=False)
    agg = summary.groupby("set")[["accuracy", "fpr", "fnr", "auc"]].agg(["mean", "std"])
    agg.to_csv(out / "summary.csv")
    print(f"=== mean (std) over seeds {args.seeds}")
    print(agg.round(3).to_string())
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
