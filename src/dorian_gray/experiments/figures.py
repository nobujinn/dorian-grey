"""Thesis figures from the latest experiment runs (vector PDF for pdflatex, PNG for previews).

Usage: uv run python -m dorian_gray.experiments.figures   -> runs/figures/
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dorian_gray.config import RUNS_DIR

# reference categorical palette, slots 1-3 (validated all-pairs); markers keep the series apart in grayscale print
STYLE = {"a": {"color": "#2a78d6", "marker": "o", "label": "Method A (forensic, LR)"},
         "b": {"color": "#eb6834", "marker": "s", "label": "Method B (CLIP probe)"},
         "a_gbt": {"color": "#1baf7a", "marker": "^", "label": "Method A (forensic, trees)"}}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
JPEG = [("clean", "none"), ("jpeg95", "95"), ("jpeg85", "85"), ("jpeg75", "75"), ("jpeg50", "50")]
RESIZE = [("resize0.5", 0.5), ("resize0.75", 0.75), ("clean", 1.0), ("resize1.5", 1.5)]
UNSEEN_OPS = [("clean", "clean"), ("blur1", "blur\nσ = 1"), ("noise3", "noise\nσ = 3"), ("webp75", "WebP\nq75"),
              ("social", "×0.75 +\nJPEG q80")]


def latest(pattern: str, threshold: str = "0.5") -> pd.DataFrame | None:
    runs = sorted(RUNS_DIR.glob(pattern))
    if not runs:
        return None
    d = pd.read_csv(runs[-1])
    return d[d.threshold == threshold] if "threshold" in d else d  # older runs: 0.5 only


def _style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def robustness_figure(path, series: list[dict], sets: list[tuple[str, str]], line_legend: list[tuple[str, str]]):
    """2x2 small multiples: rows FPR / FNR, columns JPEG quality / resize factor.

    series: one line family per entry {"pattern", "method", "linestyle" (optional)};
    sets: (test set, linestyle) for FNR; FPR only depends on the real images, shared by both sets.
    """
    loaded = [(s, latest(s["pattern"])) for s in series]
    loaded = [(s, d) for s, d in loaded if d is not None]
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.6), sharey="row", constrained_layout=True)
    for col, (conds, xlabel) in enumerate([(JPEG, "JPEG quality"), (RESIZE, "resize factor")]):
        xs = list(range(len(conds))) if conds is JPEG else [x for _, x in conds]
        for row, metric in enumerate(["fpr", "fnr"]):
            ax = axes[row, col]
            for s, d in loaded:
                v = d.set_index(["condition", "set"])[metric]
                for test_set, ls in (sets[:1] if metric == "fpr" else sets):
                    ax.plot(xs, [v[(c, test_set)] for c, _ in conds], linestyle=s.get("linestyle", ls),
                            linewidth=1.5, marker=STYLE[s["method"]]["marker"], markersize=5,
                            color=STYLE[s["method"]]["color"], markeredgecolor="white", markeredgewidth=0.8)
            ax.set_xticks(xs, [str(x) for _, x in conds])
            if row == 1:
                ax.set_xlabel(xlabel, fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel({"fpr": "FPR (real flagged fake)", "fnr": "FNR (fake passed as real)"}[metric],
                              fontsize=9, color=INK)
            _style_axes(ax)
    for ax in axes[:, 0]:  # after all panels are drawn, so the shared limits cover both columns
        ax.set_ylim(bottom=0)
    methods = dict.fromkeys(s["method"] for s, _ in loaded)
    handles = [plt.Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], label=STYLE[m]["label"])
               for m in methods]
    handles += [plt.Line2D([], [], color=MUTED, linestyle=ls, label=label) for label, ls in line_legend]
    long = sum(len(h.get_label()) for h in handles) > 70  # wrap long legends to two rows instead of clipping
    fig.legend(handles=handles, loc="outside upper center", ncol=2 if long else len(handles), frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def operations_figure(path, series: list[dict], threshold: str):
    """Lollipops per operation not seen in training: FPR and FNR on unseen generators, one mark per method."""
    loaded = [(s, latest(s["pattern"], threshold)) for s in series]
    loaded = [(s, d) for s, d in loaded if d is not None and set(c for c, _ in UNSEEN_OPS) <= set(d.condition)]
    if not loaded:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9), constrained_layout=True)
    xs = np.arange(len(UNSEEN_OPS))
    for ax, metric, ylabel in zip(axes, ["fpr", "fnr"], ["FPR (real flagged fake)", "FNR (unseen generators)"]):
        for k, (s, d) in enumerate(loaded):
            v = d[d.set == "unseen"].set_index("condition")[metric]
            x, y = xs + (k - (len(loaded) - 1) / 2) * 0.22, [v[c] for c, _ in UNSEEN_OPS]
            st = STYLE[s["method"]]
            ax.vlines(x, 0, y, color=st["color"], linewidth=2)
            ax.plot(x, y, linestyle="none", marker=st["marker"], markersize=6, color=st["color"],
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xticks(xs, [label for _, label in UNSEEN_OPS])
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
        ax.set_ylim(0, 1)  # both are rates; a common scale keeps the panels comparable
        _style_axes(ax)
    handles = [plt.Line2D([], [], color=STYLE[s["method"]]["color"], marker=STYLE[s["method"]]["marker"],
                          label=STYLE[s["method"]]["label"]) for s, _ in loaded]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def main():
    out_dir = RUNS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    # A vs. B, classifiers trained on clean images; seen and unseen generators
    robustness_figure(out_dir / "robustness", [{"pattern": f"robustness_{m}/*/metrics.csv", "method": m} for m in ("a", "b")],
                      [("seen", "-"), ("unseen", "--")], [("seen generators", "-"), ("unseen generators", "--")])
    # effect of training-time augmentation; unseen generators (the harder test set)
    robustness_figure(out_dir / "robustness_augmentation",
                      [{"pattern": f"robustness_{m}{aug}/*/metrics.csv", "method": m, "linestyle": ls}
                       for m in ("a", "b") for aug, ls in [("", "-"), ("_aug", "--")]],
                      [("unseen", "-")], [("trained on clean images", "-"), ("trained with augmentation", "--")])
    # Method A's classifier (linear vs. boosted trees) next to B; all trained with augmentation
    robustness_figure(out_dir / "robustness_classifier",
                      [{"pattern": f"robustness_{m}_aug/*/metrics.csv", "method": m} for m in ("a", "a_gbt", "b")],
                      [("seen", "-"), ("unseen", "--")], [("seen generators", "-"), ("unseen generators", "--")])
    # operations never used in training; all trained with augmentation, threshold at 5 % validation FPR
    operations_figure(out_dir / "robustness_unseen_ops",
                      [{"pattern": f"robustness_{m}_aug/*/metrics.csv", "method": m} for m in ("a", "a_gbt", "b")],
                      "val_fpr5")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
