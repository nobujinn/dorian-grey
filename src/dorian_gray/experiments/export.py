"""Copy the summary tables and figures of the latest run of every experiment into `results/` (committed).

Only aggregate numbers are exported: per-image predictions and trained models stay in `runs/`
(git-ignored), since they list dataset file names and are reproducible from the code.

Usage: uv run python -m dorian_gray.experiments.export
"""

import shutil

from dorian_gray.config import REPO_ROOT, RUNS_DIR

RESULTS_DIR = REPO_ROOT / "results"
KEEP = {".csv", ".json", ".pdf", ".png"}
SKIP = {"method_b_prototype"}  # pre-refactor runs, reproduced exactly by method_b


def main():
    for experiment in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name not in SKIP):
        runs = sorted(p for p in experiment.iterdir() if p.is_dir())
        source = experiment if experiment.name == "figures" else (runs[-1] if runs else None)
        if source is None:
            continue
        target = RESULTS_DIR / experiment.name
        shutil.rmtree(target, ignore_errors=True)
        for f in source.rglob("*"):
            if f.is_file() and f.suffix in KEEP:
                dest = target / f.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
        print(f"{experiment.name}: {source.name} -> {target.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
