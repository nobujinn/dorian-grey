"""Framework API: one image in, a score and a human-readable explanation out.

    from dorian_gray.detector import detect
    report = detect("photo.jpg")          # dict, JSON-serialisable

Method B (CLIP probe) gives the decision: it is the more reliable detector in every experiment
(results log sections 4, 6, 8, 9). Method A (forensic features, boosted trees) explains it: TreeSHAP
contributions summed per feature group say which measurable properties look synthetic or camera-like.
Default models (results log 14, 16; supervisor question 7): every image re-encoded as JPEG q75 before feature
extraction (removes the file-format shortcut), trained on Community Forensics + NTIRE training data with
augmentation; threshold for a 5 % false-positive rate on the pooled validation reals.
Scores are not calibrated probabilities yet (calibration is a planned step), and the report says so.

The models come from experiment runs and are copied to models/ (git-ignored) by `export_models`.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from dorian_gray.config import REPO_ROOT, RUNS_DIR

MODELS_DIR = REPO_ROOT / "models"
# experiment folder [/ path of the method inside its latest run]; `classify` runs keep seeds at the top level
DEFAULT_RUNS = {"a": "mixed_ntire2026_train_aug_norm-jpeg75/a_gbt", "b": "mixed_ntire2026_train_aug_norm-jpeg75/b"}

# plain-language names of Method A's feature groups (methods/forensic.py)
GROUP_TEXT = {
    "spectrum": "frequency spectrum (how fine detail fades out)",
    "periodic": "periodic upsampling patterns",
    "residual": "sensor-noise texture",
    "dct": "8x8 block / compression structure",
}


def export_models(runs: dict[str, str] = DEFAULT_RUNS, seed: int = 0, out_dir: Path = MODELS_DIR) -> None:
    """Copy the seed-`seed` model of the latest run of each method, with its validation threshold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for method, tag in runs.items():
        experiment, _, inner = tag.partition("/")
        run = sorted(p for p in (RUNS_DIR / experiment).iterdir() if p.is_dir())[-1]
        src = run / inner / f"seed{seed}"
        info = json.loads((src / "metrics.json").read_text())
        shutil.copy2(src / "model.joblib", out_dir / f"method_{method}.joblib")
        meta[method] = {"run": src.relative_to(RUNS_DIR).as_posix(), "threshold": info["val_threshold"],
                        "val_fpr_target": 0.05, "input": info["args"].get("input")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _verdict(score: float, threshold: float) -> str:
    return "synthetic" if score >= threshold else "real"


class Detector:
    """Loads the exported models once; call it on an image path or a PIL image."""

    def __init__(self, methods: tuple[str, ...] = ("a", "b"), models_dir: Path = MODELS_DIR):
        import joblib
        self.meta = json.loads((models_dir / "meta.json").read_text())
        self.models = {m: joblib.load(models_dir / f"method_{m}.joblib") for m in methods}
        self._extractors = {}

    def _extract(self, method: str, image: Image.Image) -> np.ndarray:
        """Features of the image after the model's input normalisation (e.g. JPEG q75 re-encoding)."""
        if method not in self._extractors:
            from dorian_gray.methods import METHODS
            self._extractors[method] = METHODS[method].extractor()
        if self.meta[method].get("input"):
            from dorian_gray.perturb import apply
            image = apply(image, self.meta[method]["input"])
        return self._extractors[method]([image]).astype(np.float32)

    def _method_a(self, image: Image.Image) -> dict:
        from dorian_gray.methods.forensic import contributions, explain_groups, feature_names
        x = self._extract("a", image)
        model, names = self.models["a"], feature_names()
        score = float(model.predict_proba(x)[0, 1])
        contrib = contributions(model, x)[0]
        groups = {g: float(v) for g, v in explain_groups(contrib, names).items()}
        top = np.argsort(-np.abs(contrib))[:5]
        return {"score": score, "verdict": _verdict(score, self.meta["a"]["threshold"]),
                "threshold": self.meta["a"]["threshold"],
                "groups": groups, "top_features": [(names[i], float(contrib[i])) for i in top]}

    def _method_b(self, image: Image.Image) -> dict:
        score = float(self.models["b"].predict_proba(self._extract("b", image))[0, 1])
        return {"score": score, "verdict": _verdict(score, self.meta["b"]["threshold"]),
                "threshold": self.meta["b"]["threshold"]}

    def __call__(self, image) -> dict:
        name = str(image) if not isinstance(image, Image.Image) else "<image>"
        im = image if isinstance(image, Image.Image) else Image.open(image)
        im.load()
        report = {"image": name, "format": im.format, "size": list(im.size), "methods": {}}
        if "b" in self.models:
            report["methods"]["b"] = self._method_b(im)
        if "a" in self.models:
            report["methods"]["a"] = self._method_a(im)
        main = report["methods"].get("b") or report["methods"]["a"]
        report["verdict"], report["score"] = main["verdict"], main["score"]
        report["score_note"] = "uncalibrated score in [0, 1]; threshold set for 5 % false positives on validation data"
        report["normalised"] = any(self.meta[m].get("input") for m in self.models)
        report["explanation"] = explain_text(report)
        return report


def explain_text(report: dict) -> list[str]:
    """Human-readable lines: decision, agreement of the methods, Method A's evidence per feature group."""
    lines, ms = [], report["methods"]
    decider = "Method B (CLIP probe)" if "b" in ms else "Method A (forensic features)"
    lines.append(f"Verdict: {report['verdict'].upper()} (score {report['score']:.2f}, decided by {decider}).")
    if "a" in ms and "b" in ms:
        agree = ms["a"]["verdict"] == ms["b"]["verdict"]
        lines.append("Both methods agree." if agree else
                     f"The methods disagree: Method A says {ms['a']['verdict']} (score {ms['a']['score']:.2f}).")
    if "a" in ms:
        lines.append("Evidence from measurable image properties (Method A, contribution to the log-odds):")
        for g, v in sorted(ms["a"]["groups"].items(), key=lambda kv: -abs(kv[1])):
            strength = "strongly" if abs(v) >= 1.5 else "slightly" if abs(v) < 0.5 else "clearly"
            side = "synthetic" if v > 0 else "camera-like"
            lines.append(f"  - {GROUP_TEXT[g]}: {strength} {side} ({v:+.2f})")
    if report.get("format") == "JPEG" and not report.get("normalised"):
        lines.append("Note: the image is JPEG-compressed. Compression hides generator traces, so Method A "
                     "tends to call JPEG fakes real (results log sections 3.2, 7.2).")
    return lines


_default = None


def detect(image, methods: tuple[str, ...] = ("a", "b")) -> dict:
    """Convenience wrapper with a cached Detector."""
    global _default
    if _default is None or set(_default.models) != set(methods):
        _default = Detector(methods)
    return _default(image)
