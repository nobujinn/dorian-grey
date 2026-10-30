"""Global view of Method A's per-image explanations: which feature groups push which images to "fake".

Takes a trained Method A model (latest `classify` run, one seed) and computes the per-feature
contributions to the log-odds of the test images (`forensic.contributions`: coefficient x standardised
value for logistic regression, TreeSHAP for boosted trees). Contributions are summed per feature group
and averaged per group of images: real JPEG, real PNG, fakes of seen generators, fakes of unseen
generators, and the JPEG generator LFM (unseen in seeds 0, 2, 4). So the table answers e.g. "why are
LFM fakes called real?" with the same numbers a single image's explanation would show.

Usage: uv run python -m dorian_gray.experiments.explain --run method_a_gbt_aug [--seed 0] [--n 1000]
"""

import argparse
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from dorian_gray.config import RUNS_DIR
from dorian_gray.experiments.classify import features_matrix, load_split
from dorian_gray.methods.forensic import GROUPS, contributions, explain_groups, feature_group, feature_names


def image_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    test_real = (df.part == "test_seen") & (df.label == 0)
    lfm = df.model_name == "LFM"
    return {"real JPEG": test_real & (df.format == "JPEG"), "real PNG": test_real & (df.format == "PNG"),
            "fake seen": (df.part == "test_seen") & (df.label == 1) & ~lfm,
            "fake unseen": (df.part == "test_unseen") & ~lfm,
            "LFM (JPEG fakes)": df.part.isin(["test_seen", "test_unseen"]) & lfm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="method_a_gbt_aug", help="experiment folder of a Method A `classify` run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000, help="images per image group (TreeSHAP cost)")
    args = ap.parse_args()

    run = sorted(p for p in (RUNS_DIR / args.run).iterdir() if p.is_dir())[-1]
    run_args = json.loads((run / f"seed{args.seed}" / "metrics.json").read_text())["args"]
    assert run_args["method"].startswith("a") and len(run_args["groups"]) == len(GROUPS), "full Method A runs only"
    model = joblib.load(run / f"seed{args.seed}" / "model.joblib")
    df = load_split("forensic_v1", run_args["holdout"], args.seed)
    names = feature_names()

    rng = np.random.default_rng(0)
    rows, per_feature = [], []
    for name, mask in image_groups(df).items():
        idx = df.index[mask].to_numpy()
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(args.n, len(idx)), replace=False)
        X = features_matrix(df.loc[idx])
        c = contributions(model, X)
        log_odds = model.decision_function(X)
        rows.append({"images": name, "n": len(idx), "flagged_fake": float((model.predict_proba(X)[:, 1] >= 0.5).mean()),
                     "log_odds": float(log_odds.mean()), "baseline": float((log_odds - c.sum(1)).mean()),
                     **{g: float(v.mean()) for g, v in explain_groups(c, names).items()}})
        per_feature.append(pd.DataFrame({"images": name, "feature": names, "group": [feature_group(n) for n in names],
                                         "mean": c.mean(0), "mean_abs": np.abs(c).mean(0)}))
    table = pd.DataFrame(rows)
    features = pd.concat(per_feature)

    out = RUNS_DIR / f"explain_{args.run}" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    (out / "args.json").write_text(json.dumps({**vars(args), "model_run": run.name}, indent=2))
    table.to_csv(out / "group_contributions.csv", index=False)
    features.to_csv(out / "feature_contributions.csv", index=False)

    print(f"model: {args.run}/{run.name}/seed{args.seed}; mean contribution to the log-odds of 'fake' per feature group")
    print(table.round(3).to_string(index=False))
    for name in table["images"]:
        top = features[features["images"] == name].reindex(columns=["feature", "mean"])
        top = top.iloc[np.argsort(-np.abs(top["mean"].to_numpy()))[:6]]
        print(f"--- {name}: " + ", ".join(f"{f} {v:+.2f}" for f, v in top.itertuples(index=False)))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
