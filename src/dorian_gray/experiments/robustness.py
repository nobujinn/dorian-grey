"""Robustness of method A or B to JPEG recompression and resizing (seed-0 split of `classify`).

A fixed, seeded subset of test images (n real, n fakes of seen generators, n fakes of unseen
generators) is perturbed with every condition in `perturb.CONDITIONS`, re-embedded, and scored by
the classifier trained on clean images. The threshold stays at 0.5, so a shift of the score
distribution shows up directly as a change in FPR / FNR.

Features are cached per shard with all conditions in one file, so an interrupted run resumes.
Usage: uv run python -m dorian_gray.experiments.robustness --method {a,b} [--n 1000]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dorian_gray.config import CACHE_DIR, RUNS_DIR
from dorian_gray.data.community_forensics import LOCAL_DIR
from dorian_gray.experiments.classify import features_matrix, fit_probe, load_split
from dorian_gray.features import iter_shard
from dorian_gray.methods import METHODS
from dorian_gray.metrics import binary_metrics
from dorian_gray.perturb import CONDITIONS, apply

PERTURBED = [c for c in CONDITIONS if c != "clean"]


def robust_subset(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """n images from each test group, drawn once with a fixed seed."""
    rng = np.random.default_rng(seed)
    groups = {"real": (df.part == "test_seen") & (df.label == 0),
              "fake_seen": (df.part == "test_seen") & (df.label == 1),
              "fake_unseen": df.part == "test_unseen"}
    picks = []
    for name, mask in groups.items():
        idx = df.index[mask].to_numpy()
        picks.append(df.loc[rng.choice(idx, min(n, len(idx)), replace=False)].assign(group=name))
    return pd.concat(picks)


def extract(subset: pd.DataFrame, embedder, cache: str, dtype: str, batch_size: int = 32) -> None:
    """Read each shard once and embed every perturbed version of the selected images."""
    for shard, g in sorted(subset.groupby("shard")):
        out = CACHE_DIR / "features" / cache / f"{Path(shard).stem}.npz"
        rows = np.sort(g.row.to_numpy())
        if out.exists() and np.array_equal(np.load(out)["row"], rows):
            continue
        feats = {c: [] for c in PERTURBED}
        with tqdm(total=len(rows), desc=Path(shard).stem) as bar:
            for _, images in iter_shard(LOCAL_DIR / "data" / shard, rows, batch_size):
                for c in PERTURBED:
                    feats[c].append(embedder([apply(im, c) for im in images]).astype(dtype))
                bar.update(len(images))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, row=rows, **{c: np.concatenate(v) for c, v in feats.items()})


def load_robust(subset: pd.DataFrame, cache: str) -> pd.DataFrame:
    """The subset with one feature column per condition (`clean` = the original cached features)."""
    frames = []
    for shard in sorted(subset.shard.unique()):
        z = np.load(CACHE_DIR / "features" / cache / f"{Path(shard).stem}.npz")
        frames.append(pd.DataFrame({"shard": shard, "row": z["row"], **{c: list(z[c]) for c in PERTURBED}}))
    return subset.rename(columns={"features": "clean"}).merge(pd.concat(frames), on=["shard", "row"], validate="1:1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--n", type=int, default=1000, help="images per group (real / fake seen / fake unseen)")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    method = METHODS[args.method]
    cache = f"{method.features}_robust"
    df = load_split(method.features, args.holdout, args.seed)
    probe = fit_probe(df, method.classifier(args.C))
    subset = robust_subset(df, args.n, args.seed)
    extract(subset, method.extractor(), cache, method.cache_dtype)
    data = load_robust(subset, cache)

    metrics, by_format = [], []
    for c in CONDITIONS:
        data["p_fake"] = probe.predict_proba(features_matrix(data.rename(columns={c: "features"})))[:, 1]
        for name, fakes in [("seen", "fake_seen"), ("unseen", "fake_unseen")]:
            m = data.group.isin(["real", fakes])
            metrics.append({"condition": c, "set": name, **binary_metrics(data.label[m], data.p_fake[m])})
        # did recompression erase the format difference between real JPEG and real PNG?
        flagged = data.assign(flag=data.p_fake >= 0.5).groupby(["group", "format"]).flag.agg(["size", "mean"])
        by_format.append(flagged.reset_index().assign(condition=c))
    metrics, by_format = pd.DataFrame(metrics), pd.concat(by_format)

    out = RUNS_DIR / f"robustness_{args.method}" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    metrics.to_csv(out / "metrics.csv", index=False)
    by_format.to_csv(out / "flagged_by_format.csv", index=False)

    order = list(CONDITIONS)
    for key in ["accuracy", "fpr", "fnr", "auc"]:
        print(f"--- {key}")
        print(metrics.pivot(index="condition", columns="set", values=key).loc[order].round(3).to_string())
    print("--- flagged as fake, by group and original format")
    print(by_format.pivot_table(index="condition", columns=["group", "format"], values="mean").loc[order].round(3).to_string())
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
