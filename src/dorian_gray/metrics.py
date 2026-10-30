"""Evaluation metrics. Label convention everywhere: 1 = synthetic/fake (positive), 0 = real."""

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_metrics(y_true: np.ndarray, p_fake: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy, FPR, FNR at a fixed threshold, plus threshold-free AUC.

    FPR = real images flagged as fake / all real images.
    FNR = fake images passed as real / all fake images.
    The threshold is fixed in advance (not re-tuned per test set), so a drifting score
    distribution, e.g. after JPEG compression, shows up as a change in FPR/FNR.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(p_fake) >= threshold).astype(int)
    real, fake = y_true == 0, y_true == 1
    out = {
        "n_real": int(real.sum()),
        "n_fake": int(fake.sum()),
        "accuracy": float((y_pred == y_true).mean()),
        "fpr": float(y_pred[real].mean()) if real.any() else float("nan"),
        "fnr": float(1 - y_pred[fake].mean()) if fake.any() else float("nan"),
    }
    out["auc"] = float(roc_auc_score(y_true, p_fake)) if real.any() and fake.any() else float("nan")
    return out


def threshold_at_fpr(p_real: np.ndarray, target: float) -> float:
    """Smallest threshold at which at most `target` of the given real images are flagged fake.

    Chosen on validation reals only, so test sets never influence it. Compared with the fixed 0.5,
    it puts every method at the same false-positive rate, the operating point a deployed detector
    would be set to; FNR at that point is then directly comparable between methods.
    """
    p = np.sort(np.asarray(p_real, dtype=float))[::-1]
    k = int(np.floor(target * len(p)))  # number of reals allowed above the threshold
    return float(np.nextafter(p[k], np.inf)) if k < len(p) else float(p[-1])
