"""FaceForensics++ (Rössler et al., ICCV 2019), c23 compression, as face crops for the face-swap experiments.

The videos (Kaggle copy `xdxd003/ff-c23`) live in DATASETS_DIR/ff-c23/FaceForensics++_C23/<method>/*.mp4.
Protocol, as in the FF++ paper and most later work:
  * the official split by video pair (splits/{train,val,test}.json from the FaceForensics repository,
    720 / 140 / 140 pairs). Original video `a` is in the split of its pair; the fakes `a_b` and `b_a`
    (target a, source b and vice versa) are in the same split, so no identity crosses splits;
  * `FRAMES` frames per video, evenly spaced; the most confident face (YuNet detector, OpenCV) is cropped
    as a square 1.3x the detection box, so some context around the face is kept;
  * every crop is saved as **PNG**: real and fake faces are processed identically (same decoder, same crop,
    no re-compression), so there is no file-format difference between the classes (cf. results log 3.2).
DeepFakeDetection is skipped: its real videos are not in this copy.

Usage: uv run python -m dorian_gray.data.faceforensics [--frames 8] [--methods Deepfakes ...]
Writes DATASETS_DIR/ff-c23-faces/<method>/<video>_<frame>.png and manifest.csv (path,label,source,video,split).
"""

import argparse
import json
import urllib.request

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from dorian_gray.config import DATASETS_DIR

VIDEO_DIR = DATASETS_DIR / "ff-c23" / "FaceForensics++_C23"
SPLIT_DIR = DATASETS_DIR / "ff-c23" / "splits"
FACES_DIR = DATASETS_DIR / "ff-c23-faces"
METHODS = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
FRAMES = 8
SCALE = 1.3
MIN_SCORE = 0.8
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_PATH = DATASETS_DIR / "models" / "face_detection_yunet_2023mar.onnx"


def video_splits() -> dict[str, str]:
    """Video id ('071' for originals, '071_054' for fakes) -> 'train' / 'val' / 'test'."""
    out = {}
    for split in ("train", "val", "test"):
        for a, b in json.loads((SPLIT_DIR / f"{split}.json").read_text()):
            out.update({a: split, b: split, f"{a}_{b}": split, f"{b}_{a}": split})
    return out


def detector() -> "cv2.FaceDetectorYN":
    if not YUNET_PATH.exists():
        YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score_threshold=MIN_SCORE)


def square_box(x: float, y: float, w: float, h: float, width: int, height: int, scale: float = SCALE):
    """Square crop around a detection box, `scale` times its longer side, shifted to stay inside the frame."""
    side = min(int(round(max(w, h) * scale)), width, height)
    x0 = int(round(x + w / 2 - side / 2))
    y0 = int(round(y + h / 2 - side / 2))
    x0, y0 = min(max(x0, 0), width - side), min(max(y0, 0), height - side)
    return x0, y0, side


def face_crops(path, det, frames: int = FRAMES) -> list[tuple[int, np.ndarray]]:
    """(frame index, RGB crop) of the most confident face in `frames` evenly spaced frames."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, n - 1, frames + 2)[1:-1].astype(int):  # skip the very first / last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(frame)
        if faces is None or not len(faces):
            continue
        x, y, bw, bh = faces[np.argmax(faces[:, -1]), :4]
        x0, y0, side = square_box(x, y, bw, bh, w, h)
        out.append((int(i), cv2.cvtColor(frame[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


def readable(path) -> bool:
    """False for a file cut short by an interrupted run."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.load()
        return True
    except OSError:
        return False


def build(methods: list[str] = METHODS, frames: int = FRAMES) -> pd.DataFrame:
    from PIL import Image

    splits, det, rows = video_splits(), detector(), []
    for method in methods:
        (FACES_DIR / method).mkdir(parents=True, exist_ok=True)
        for video in tqdm(sorted((VIDEO_DIR / method).glob("*.mp4")), desc=method):
            done = sorted((FACES_DIR / method).glob(f"{video.stem}_[0-9][0-9][0-9][0-9].png"))
            if len(done) == frames and all(map(readable, done)):  # finished in an earlier (interrupted) run
                rels = [f"{method}/{p.name}" for p in done]
            else:
                rels = []
                for i, crop in face_crops(video, det, frames):
                    rels.append(f"{method}/{video.stem}_{i:04d}.png")
                    Image.fromarray(crop).save(FACES_DIR / rels[-1])
            rows += [{"path": rel, "label": int(method != "original"), "source": method,
                      "video": video.stem, "split": splits[video.stem]} for rel in rels]
    df = pd.DataFrame(rows)
    df.to_csv(FACES_DIR / "manifest.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    args = ap.parse_args()
    df = build(args.methods, args.frames)
    print(df.groupby(["split", "source"]).agg(videos=("video", "nunique"), crops=("path", "size")).unstack(0))


if __name__ == "__main__":
    main()
