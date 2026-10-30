import numpy as np
import pytest
from PIL import Image

from dorian_gray.methods.forensic import (GROUPS, ForensicFeatures, explain, feature_group,
                                          image_features, make_boosting, make_classifier)
from dorian_gray.perturb import jpeg


def _smooth_noise(size=320, seed=0):
    """Natural-ish image: low-pass filtered noise (1/f-like spectrum), not white noise."""
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    x = ndimage.gaussian_filter(rng.normal(size=(size, size, 3)), (2, 2, 0)) * 400 + 128
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


def test_all_features_finite_and_grouped():
    f = image_features(_smooth_noise())
    assert len(f) > 100 and np.isfinite(list(f.values())).all()
    assert {feature_group(k) for k in f} == set(GROUPS)


def test_small_and_odd_sized_images():
    assert np.isfinite(ForensicFeatures()([_smooth_noise(50), _smooth_noise(131)])).all()


def test_periodic_pattern_gives_peak():
    im = np.asarray(_smooth_noise(), float)
    pattern = np.tile([6, 0, -6, 0], (320, 80))[..., None]  # horizontal period 4
    periodic = Image.fromarray(np.clip(im + pattern, 0, 255).astype(np.uint8))
    clean, marked = image_features(_smooth_noise()), image_features(periodic)
    assert marked["peak_x_2of8"] > clean["peak_x_2of8"] + 2  # period 4 -> 2/8 cycle/pixel


def test_jpeg_raises_blockiness():
    im = _smooth_noise()
    assert image_features(jpeg(im, 50))["blockiness_h"] > image_features(im)["blockiness_h"] + 0.1


def test_explain_returns_top_contributions():
    ff = ForensicFeatures()
    X = ff([_smooth_noise(seed=s) for s in range(6)] + [jpeg(_smooth_noise(seed=s), 40) for s in range(6)])
    clf = make_classifier().fit(X, [0] * 6 + [1] * 6)
    top = explain(clf, X[-1], ff.names, k=3)
    assert len(top) == 3 and all(name in ff.names for name, _ in top)
    assert abs(top[0][1]) >= abs(top[-1][1])


def test_boosting_learns_interaction_linear_model_cannot():
    # label = sign(x0 * x1): no single feature is informative, only their combination
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    tr, te = slice(0, 3000), slice(3000, None)
    acc = lambda clf: (clf.fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean()
    assert acc(make_classifier()) < 0.6 < 0.9 < acc(make_boosting())
