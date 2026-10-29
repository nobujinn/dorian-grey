"""Thesis figures from the latest experiment runs (vector PDF for pdflatex, PNG for previews).

Usage: uv run python -m dorian_gray.experiments.figures   -> runs/figures/
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dorian_gray.config import RUNS_DIR

# reference categorical palette, slots 1-2; markers keep the methods apart in grayscale print
STYLE = {"a": {"color": "#2a78d6", "marker": "o", "label": "Method A (forensic)"},
         "b": {"color": "#eb6834", "marker": "s", "label": "Method B (CLIP probe)"}}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
JPEG = [("clean", "none"), ("jpeg95", "95"), ("jpeg85", "85"), ("jpeg75", "75"), ("jpeg50", "50")]
RESIZE = [("resize0.5", 0.5), ("resize0.75", 0.75), ("clean", 1.0), ("resize1.5", 1.5)]


def latest(pattern: str) -> pd.DataFrame | None:
    runs = sorted(RUNS_DIR.glob(pattern))
    return pd.read_csv(runs[-1]) if runs else None


def _style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def robustness_figure(out_dir):
    data = {m: latest(f"robustness_{m}/*/metrics.csv") for m in STYLE}
    data = {m: d for m, d in data.items() if d is not None}
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.6), sharey="row", constrained_layout=True)
    for col, (conds, xlabel) in enumerate([(JPEG, "JPEG quality"), (RESIZE, "resize factor")]):
        xs = list(range(len(conds))) if conds is JPEG else [x for _, x in conds]
        for row, metric in enumerate(["fpr", "fnr"]):
            ax = axes[row, col]
            for m, d in data.items():
                v = d.set_index(["condition", "set"])[metric]
                # FPR only depends on the real images, which both test sets share
                sets = [("seen", "-")] if metric == "fpr" else [("seen", "-"), ("unseen", "--")]
                for s, ls in sets:
                    ax.plot(xs, [v[(c, s)] for c, _ in conds], linestyle=ls, linewidth=1.5,
                            marker=STYLE[m]["marker"], markersize=5, color=STYLE[m]["color"],
                            markeredgecolor="white", markeredgewidth=0.8)
            ax.set_xticks(xs, [str(x) for _, x in conds])
            if row == 1:
                ax.set_xlabel(xlabel, fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel({"fpr": "FPR (real flagged fake)", "fnr": "FNR (fake passed as real)"}[metric],
                              fontsize=9, color=INK)
            _style_axes(ax)
    for ax in axes[:, 0]:  # after all panels are drawn, so the shared limits cover both columns
        ax.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=s["color"], marker=s["marker"], label=s["label"])
               for m, s in STYLE.items() if m in data]
    handles += [plt.Line2D([], [], color=MUTED, linestyle="-", label="seen generators"),
                plt.Line2D([], [], color=MUTED, linestyle="--", label="unseen generators")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"robustness.{ext}", dpi=200)
    plt.close(fig)


def main():
    out_dir = RUNS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    robustness_figure(out_dir)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
