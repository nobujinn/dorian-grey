import pandas as pd

from dorian_gray.experiments.faces import metrics_rows


def test_metrics_per_frame_video_and_method():
    df = pd.DataFrame({
        "split": ["test"] * 6 + ["train"],
        "video": ["000", "000", "000_001", "000_001", "002_003", "002_003", "009"],
        "source": ["original", "original", "Deepfakes", "Deepfakes", "FaceSwap", "FaceSwap", "original"],
        "label": [0, 0, 1, 1, 1, 1, 0],
        "p_fake": [0.2, 0.6, 0.9, 0.5, 0.4, 0.3, 0.99],
    })
    rows = {(r["level"], r["subset"]): r for r in metrics_rows(df, {"0.5": 0.5}, method="b")}
    assert set(rows) == {(lv, s) for lv in ("frame", "video") for s in ("all", "Deepfakes", "FaceSwap")}
    assert rows["frame", "all"]["n_real"] == 2 and rows["frame", "all"]["n_fake"] == 4   # train rows excluded
    assert rows["video", "all"]["n_real"] == 1 and rows["video", "all"]["n_fake"] == 2
    assert rows["video", "Deepfakes"]["fnr"] == 0.0 and rows["video", "FaceSwap"]["fnr"] == 1.0  # means .7 / .35
