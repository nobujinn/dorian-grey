"""Dorian Gray — explainable detection of synthetic and manipulated images."""

__version__ = "0.1.0"


def main() -> None:
    """CLI: `dorian-gray photo.jpg [more.png ...] [--json] [--method a b]`, `dorian-gray --export-models`,
    `dorian-gray --info`."""
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="dorian-gray", description="Detect synthetic images and explain the decision.")
    ap.add_argument("images", nargs="*", help="image files to analyse")
    ap.add_argument("--method", nargs="+", choices=["a", "b"], default=["a", "b"])
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    ap.add_argument("--export-models", action="store_true", help="copy the chosen experiment models to models/")
    ap.add_argument("--info", action="store_true", help="show version, paths and device")
    args = ap.parse_args()

    if args.info or not (args.images or args.export_models):
        from dorian_gray.config import DATASETS_DIR, device
        print(f"dorian-gray {__version__}\ndatasets dir: {DATASETS_DIR}\ndevice:       {device()}")
        if not args.info:
            ap.print_usage()
        return
    from dorian_gray.detector import Detector, export_models
    if args.export_models:
        export_models()
        print("models exported to models/")
    if args.images:
        detector = Detector(tuple(args.method))
        for path in args.images:
            report = detector(path)
            print(json.dumps(report, indent=2) if args.json else f"{path}\n" + "\n".join(report["explanation"]) + "\n")
