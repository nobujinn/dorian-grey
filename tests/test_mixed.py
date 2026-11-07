import numpy as np
import pandas as pd

from dorian_gray.experiments.mixed import set_metrics, split_extra


def test_split_extra_is_seeded_and_80_10_10():
    df = pd.DataFrame({"label": [0, 1] * 500})
    a, b, c = split_extra(df, 0), split_extra(df, 0), split_extra(df, 1)
    assert a.equals(b) and not a.equals(c)
    assert a.value_counts().to_dict() == {"train": 800, "val": 100, "test": 100}


def test_set_metrics_per_source():
    df = pd.DataFrame({"label": [0, 1, 0, 1], "p_fake": [0.1, 0.9, 0.6, 0.4], "source": ["x", "x", "y", "y"]})
    rows = set_metrics(df, {"0.5": 0.5}, "d")
    by = {r["subset"]: r for r in rows}
    assert set(by) == {"all", "source=x", "source=y"}
    assert by["source=x"]["auc"] == 1.0 and by["source=y"]["auc"] == 0.0 and by["all"]["fpr"] == 0.5
