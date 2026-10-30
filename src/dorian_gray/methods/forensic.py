"""Method A: hand-crafted forensic features (frequency spectrum, noise residuals, DCT) + logistic regression.

Every feature has a name and a physical meaning, so the classifier's decision can be explained as
"which measured properties of this image pushed the score up or down". Four groups:

  * spectrum - azimuthally averaged log power spectrum of the luminance (`durall2020watch`): generators
               reproduce the high-frequency fall-off of natural images poorly;
  * periodic - prominence of spectral peaks of the noise residual at multiples of 1/8 cycle/pixel:
               upsampling layers leave periodic traces (`frank2020leveraging`, `corvi2023detection`);
               JPEG's 8x8 grid also peaks there, which the per-format analysis has to keep in mind;
  * residual - statistics and co-occurrences of high-pass noise residuals, a compact version of the
               rich models of `fridrich2012rich`; inter-channel residual correlation reflects the
               camera's demosaicing, which generated images do not have;
  * dct      - energy of each 8x8 block-DCT frequency and blockiness, i.e. JPEG compression history.

Images are never resampled (resampling itself leaves spectral traces): features are computed on a
central crop of at most 256 px, aligned to the 8x8 grid of the original image.
"""

import numpy as np
from PIL import Image
from scipy import fft, ndimage
from scipy.stats import kurtosis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CROP, MIN_CROP = 256, 64
RADIAL_BINS = 24
PEAK_STEPS = (1, 2, 3, 4)  # peaks at k/8 cycle/pixel (k = 4 is the Nyquist frequency)
COOC_T = 2                 # residuals are rounded and truncated to [-T, T] before co-occurrences
FILTERS = {
    "d1h": np.array([[-1, 1]], float),
    "d1v": np.array([[-1], [1]], float),
    "d2h": np.array([[1, -2, 1]], float),
    "d2v": np.array([[1], [-2], [1]], float),
    "kv3": np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], float) / 4,  # SRM "square 3x3"
}
COOC_FILTERS = ("d1h", "kv3")
# residual for the periodic peaks: kv3 is separable, (2-2cos wx)(2-2cos wy)/4, so it is zero on both
# frequency axes and would hide purely horizontal/vertical periodicity; the Laplacian is not
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)


def _crop(image: Image.Image) -> np.ndarray:
    """RGB float array of the central crop; side = min(256, shorter side) rounded down to a multiple of 8."""
    image = image.convert("RGB")
    if min(image.size) < MIN_CROP:  # rare tiny images: the only case that is upscaled
        s = MIN_CROP / min(image.size)
        image = image.resize((round(image.width * s), round(image.height * s)), Image.BICUBIC)
    x = np.asarray(image, dtype=np.float32)
    h, w = x.shape[:2]
    c = min(CROP, h, w) // 8 * 8
    top, left = (h - c) // 2 // 8 * 8, (w - c) // 2 // 8 * 8
    return x[top:top + c, left:left + c]


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], np.float32)


def _power(y: np.ndarray) -> np.ndarray:
    """Log power spectrum with a Hann window (suppresses the cross-shaped border leakage)."""
    n = y.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n))
    return np.log(np.abs(fft.fft2((y - y.mean()) * win)) ** 2 + 1e-8)


def _radial_freq(n: int) -> np.ndarray:
    f = fft.fftfreq(n)
    return np.hypot(f[:, None], f[None, :])


def spectrum_features(y: np.ndarray) -> dict:
    p, r = _power(y), _radial_freq(y.shape[0])
    edges = np.linspace(0, 0.5, RADIAL_BINS + 1)
    idx = np.digitize(r, edges) - 1
    ok = (idx >= 0) & (idx < RADIAL_BINS) & (r > 0)
    profile = np.bincount(idx[ok], p[ok], RADIAL_BINS) / np.maximum(np.bincount(idx[ok], minlength=RADIAL_BINS), 1)
    out = {f"spec_r{i:02d}": v for i, v in enumerate(profile - profile.mean())}  # shape only, not level
    centres = (edges[:-1] + edges[1:]) / 2
    hf = centres > 0.25
    out["spec_slope_hf"] = np.polyfit(np.log(centres[hf]), profile[hf], 1)[0]  # high-frequency power-law slope
    return out


def periodic_features(residual: np.ndarray) -> dict:
    p, n, out = _power(residual), residual.shape[0], {}
    for k in PEAK_STEPS:
        i = round(k / 8 * n) % n
        for name, (u, v) in {"x": (0, i), "y": (i, 0), "d": (i, i)}.items():
            patch = p[np.ix_(np.arange(u - 3, u + 4) % n, np.arange(v - 3, v + 4) % n)].ravel()
            ring = np.delete(patch, 24)  # 7x7 neighbourhood without the centre
            out[f"peak_{name}_{k}of8"] = p[u, v] - np.median(ring)  # log ratio peak / local background
    return out


def residual_features(rgb: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for name, kern in FILTERS.items():
        r = ndimage.correlate(y, kern, mode="reflect")
        out[f"res_{name}_logstd"] = np.log(r.std() + 1e-6)
        out[f"res_{name}_kurt"] = kurtosis(r, axis=None)
        if name in COOC_FILTERS:
            q = np.clip(np.round(r), -COOC_T, COOC_T).astype(int) + COOC_T
            pairs = q[:, :-1] * (2 * COOC_T + 1) + q[:, 1:]  # horizontal neighbour pairs
            hist = np.bincount(pairs.ravel(), minlength=(2 * COOC_T + 1) ** 2) / pairs.size
            out.update({f"cooc_{name}_{a - COOC_T}_{b - COOC_T}": hist[a * (2 * COOC_T + 1) + b]
                        for a in range(2 * COOC_T + 1) for b in range(2 * COOC_T + 1)})
    kv = [ndimage.correlate(rgb[..., ch], FILTERS["kv3"], mode="reflect").ravel() for ch in range(3)]
    for (a, b), name in zip([(0, 1), (1, 2), (0, 2)], ["rg", "gb", "rb"]):
        out[f"rgb_corr_{name}"] = np.corrcoef(kv[a], kv[b])[0, 1]
    return out


def dct_features(y: np.ndarray) -> dict:
    n = y.shape[0] // 8
    blocks = y.reshape(n, 8, n, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8) - 128
    coef = fft.dctn(blocks, axes=(1, 2), norm="ortho")
    logstd = np.log(coef.std(axis=0) + 1e-3)
    out = {f"dct_{u}{v}_logstd": logstd[u, v] for u in range(8) for v in range(8) if (u, v) != (0, 0)}
    # blockiness: mean |step| across 8x8 block borders relative to inside blocks (1 = no grid)
    for axis, name in [(1, "h"), (0, "v")]:
        d = np.abs(np.diff(y, axis=axis))
        pos = np.arange(d.shape[axis]) % 8 == 7
        across, inside = (d[:, pos], d[:, ~pos]) if axis == 1 else (d[pos], d[~pos])
        out[f"blockiness_{name}"] = across.mean() / (inside.mean() + 1e-6)
    return out


GROUPS = ("spectrum", "periodic", "residual", "dct")


def image_features(image: Image.Image) -> dict:
    rgb = _crop(image)
    y = _luma(rgb)
    residual = ndimage.correlate(y, LAPLACIAN, mode="reflect")
    return {**spectrum_features(y), **periodic_features(residual), **residual_features(rgb, y), **dct_features(y)}


def feature_names() -> list[str]:
    rng = np.random.default_rng(0)
    return list(image_features(Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))))


def feature_group(name: str) -> str:
    prefix = name.split("_")[0]
    return {"spec": "spectrum", "peak": "periodic", "res": "residual", "cooc": "residual",
            "rgb": "residual", "dct": "dct", "blockiness": "dct"}[prefix]


class ForensicFeatures:
    """Same interface as `ClipEmbedder`: list of images -> (N, D) array. CPU only."""

    def __init__(self):
        self.names = feature_names()

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        rows = [image_features(im) for im in images]
        return np.nan_to_num(np.array([[r[k] for k in self.names] for r in rows], dtype=np.float32))


def make_classifier(C: float = 1.0):
    """Standardised features + logistic regression: each coefficient is the log-odds change per
    standard deviation of one named feature, which is what the explanation reports."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def make_boosting(C: float = 1.0):
    """Gradient-boosted trees on the same features: non-linear interactions (e.g. "blockiness only
    matters when the spectrum is also flat") that the linear model cannot express. `C` is unused,
    it keeps the factory signature; trees need no scaling. Early stopping on 10 % of the training set."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=True, n_iter_no_change=20,
                                          class_weight="balanced", random_state=0)


def contributions(classifier, X: np.ndarray) -> np.ndarray:
    """Per-feature contributions to the log-odds of "fake", shape (N, D); they sum to the model's
    log-odds minus a baseline, so they can also be summed per feature group.

    Logistic regression: coef * standardised value (baseline = the average training image).
    Boosted trees: TreeSHAP values (`lundberg2020local`), exact Shapley values of the tree
    ensemble (baseline = the expected log-odds over the training data)."""
    X = np.atleast_2d(X)
    if hasattr(classifier, "steps"):
        scaler, lr = classifier[0], classifier[-1]
        return lr.coef_[0] * scaler.transform(X)
    import shap  # only needed for the tree model
    return np.asarray(shap.TreeExplainer(classifier).shap_values(X))


def explain(classifier, x: np.ndarray, names: list[str], k: int = 5) -> list[tuple[str, float]]:
    """Top-k features by contribution to the log-odds of "fake" for one image."""
    contrib = contributions(classifier, x)[0]
    order = np.argsort(-np.abs(contrib))[:k]
    return [(names[i], float(contrib[i])) for i in order]


def explain_groups(contrib: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Contributions summed per feature group (GROUPS): "the spectrum pushed towards fake by +1.2"."""
    groups = np.array([feature_group(n) for n in names])
    return {g: contrib[..., groups == g].sum(-1) for g in GROUPS}
