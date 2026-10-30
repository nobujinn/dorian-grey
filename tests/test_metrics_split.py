import numpy as np
import pandas as pd

from dorian_gray.experiments.classify import split
from dorian_gray.metrics import binary_metrics, threshold_at_fpr


def test_binary_metrics_counts():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.7, 0.4, 0.9, 0.3, 0.8, 0.6])  # 1 false positive, 1 false negative
    m = binary_metrics(y, p)
    assert (m["n_real"], m["n_fake"]) == (4, 4)
    assert m["fpr"] == 0.25 and m["fnr"] == 0.25 and m["accuracy"] == 0.75


def test_binary_metrics_single_class_has_no_auc():
    m = binary_metrics(np.ones(3), np.array([0.9, 0.2, 0.7]))
    assert np.isnan(m["fpr"]) and np.isnan(m["auc"]) and abs(m["fnr"] - 1 / 3) < 1e-9


def _toy_frame(n_real=100, gens=10, per_gen=20):
    rows = [{"label": 0, "model_name": "real"} for _ in range(n_real)]
    rows += [{"label": 1, "model_name": f"g{g}"} for g in range(gens) for _ in range(per_gen)]
    return pd.DataFrame(rows)


def test_split_holds_out_whole_generators():
    df = _toy_frame()
    part = split(df, holdout=0.3, seed=0)
    unseen = set(df.model_name[part == "test_unseen"])
    seen = set(df.model_name[(df.label == 1) & (part != "test_unseen")])
    assert len(unseen) == 3 and not unseen & seen          # no generator leaks into training
    assert (df.label[part == "test_unseen"] == 1).all()     # unseen test set contains only fakes
    assert set(part[df.label == 0]) == {"train", "val", "test_seen"}


def test_split_is_deterministic():
    df = _toy_frame()
    assert split(df, 0.3, seed=1).equals(split(df, 0.3, seed=1))


def test_threshold_at_fpr_bounds_validation_fpr():
    p = np.random.default_rng(0).random(1000)
    for target in (0.01, 0.05, 0.2):
        t = threshold_at_fpr(p, target)
        assert target - 0.002 <= (p >= t).mean() <= target
