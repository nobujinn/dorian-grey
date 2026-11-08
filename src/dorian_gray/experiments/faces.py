"""Face swap / face manipulation: Methods A and B on FaceForensics++ c23 face crops (`data.faceforensics`).

Protocols (official FF++ video split, so no identity is shared between train, val and test):
  * `all`: train on originals + all manipulation methods; test on the test split, overall and per method;
  * `loo` (leave one out): for every manipulation method, train without it and test on it, i.e. a
    manipulation never seen in training (the face-swap analogue of the unseen generators in `classify`).
Thresholds as everywhere: 0.5 and `val_fpr5` (5 % of the validation real *frames* flagged). Metrics per
frame and per video (mean score of the video's frames, same thresholds).
Features are cached in cache/features/external/ff_c23_faces/ (shared with `external` and `mixed`).

Usage: uv run python -m dorian_gray.experiments.faces [--methods a a_gbt b] [--protocols all loo]
"""

import argparse
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from dorian_gray.config import RUNS_DIR
from dorian_gray.data.faceforensics import FACES_DIR
from dorian_gray.experiments.classify import OPERATING_POINTS, TARGET_FPR, features_matrix
from dorian_gray.experiments.mixed import load_set, prefetch
from dorian_gray.methods import METHODS
from dorian_gray.metrics import binary_metrics, threshold_at_fpr

NAME = "ff_c23_faces"


def metrics_rows(df: pd.DataFrame, thresholds: dict, **keys) -> list[dict]:
    """Test-split metrics per frame and per video, for all fakes and per manipulation method."""
    test = df[df.split == "test"]
    videos = test.groupby(["video", "source"], as_index=False).agg(label=("label", "first"), p_fake=("p_fake", "mean"))
    rows = []
    for level, d in [("frame", test), ("video", videos)]:
        real = d.label == 0
        for subset in ["all"] + sorted(d.loc[~real, "source"].unique()):
            m = real | ((d.source == subset) if subset != "all" else ~real)
            rows += [{**keys, "level": level, "threshold": op, "subset": subset,
                      **binary_metrics(d.label[m], d.p_fake[m], t)} for op, t in thresholds.items()]
    return rows


def fit_and_score(df: pd.DataFrame, method_key: str, C: float, train_mask: pd.Series):
    model = METHODS[method_key].classifier(C).fit(features_matrix(df[train_mask]), df.label[train_mask])
    p = model.predict_proba(features_matrix(df))[:, 1]
    val_real = ((df.split == "val") & (df.label == 0)).to_numpy()
    thresholds = dict(zip(OPERATING_POINTS, [0.5, threshold_at_fpr(p[val_real], TARGET_FPR)]))
    return model, p, thresholds


def run(args, method_key: str, out) -> list[dict]:
    df = load_set(NAME, str(FACES_DIR), "manifest.csv", METHODS[method_key].features)  # keeps video, split
    manipulations = sorted(df.loc[df.label == 1, "source"].unique())
    rows = []
    for protocol in args.protocols:
        held_out = manipulations if protocol == "loo" else [None]
        for m in held_out:
            train = (df.split == "train") & (df.source != m)
            model, p, thresholds = fit_and_score(df, method_key, args.C, train)
            d = df.assign(p_fake=p)
            if m is not None:  # evaluate only on the held-out manipulation (+ test reals)
                d = d[(d.label == 0) | (d.source == m)]
            rows += metrics_rows(d, thresholds, method=method_key, protocol=protocol, held_out=m or "-")
            if protocol == "all":
                (out / method_key).mkdir(parents=True, exist_ok=True)
                joblib.dump(model, out / method_key / "model.joblib")
                (out / method_key / "metrics.json").write_text(json.dumps(
                    {"args": vars(args), "val_threshold": thresholds["val_fpr5"]}, indent=2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=["a", "a_gbt", "b"])
    ap.add_argument("--protocols", nargs="+", choices=["all", "loo"], default=["all", "loo"])
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    prefetch([(NAME, str(FACES_DIR), "manifest.csv")], sorted({METHODS[m].features for m in args.methods}))
    out = RUNS_DIR / "faces_ff_c23" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    rows = pd.DataFrame([r for m in args.methods for r in run(args, m, out)])
    rows.to_csv(out / "metrics.csv", index=False)
    show = rows[(rows.threshold == "val_fpr5") & ((rows.subset == "all") | (rows.protocol == "loo"))]
    show = show[(show.protocol == "all") | (show.subset == show.held_out)]
    print(show[["method", "protocol", "held_out", "level", "subset", "fpr", "fnr", "auc"]].round(3).to_string(index=False))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
