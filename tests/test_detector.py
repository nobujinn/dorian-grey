import json

import numpy as np
import pytest
from PIL import Image

from dorian_gray import detector
from dorian_gray.methods.forensic import feature_names, make_boosting


@pytest.fixture
def models_dir(tmp_path):
    """A tiny Method A model on random features, exported like the real ones."""
    import joblib
    pytest.importorskip("shap")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(feature_names())))
    joblib.dump(make_boosting().fit(X, (X[:, 0] > 0).astype(int)), tmp_path / "method_a.joblib")
    (tmp_path / "meta.json").write_text(json.dumps({"a": {"threshold": 0.5}}))
    return tmp_path


def test_detector_report_and_explanation(models_dir):
    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8))
    report = detector.Detector(("a",), models_dir)(image)
    a = report["methods"]["a"]
    assert 0 <= report["score"] <= 1 and report["verdict"] in {"real", "synthetic"}
    assert set(a["groups"]) == {"spectrum", "periodic", "residual", "dct"} and len(a["top_features"]) == 5
    assert report["explanation"][0].startswith("Verdict:")
    assert sum(line.startswith("  - ") for line in report["explanation"]) == 4
    json.dumps(report)  # the report must be JSON-serialisable
