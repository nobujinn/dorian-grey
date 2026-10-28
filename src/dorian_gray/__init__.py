"""Dorian Gray — explainable detection of synthetic and manipulated images."""

__version__ = "0.1.0"


def main() -> None:
    from dorian_gray.config import DATASETS_DIR, device

    print(f"dorian-gray {__version__}")
    print(f"datasets dir: {DATASETS_DIR}")
    print(f"device:       {device()}")
