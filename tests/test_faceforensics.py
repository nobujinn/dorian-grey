import json

from dorian_gray.data import faceforensics as ff


def test_square_box_scaled_and_inside_frame():
    x0, y0, side = ff.square_box(100, 100, 40, 50, 640, 480)
    assert side == 65 and (x0, y0) == (88, 92)          # 1.3 x longer side, centred on the box
    x0, y0, side = ff.square_box(620, 460, 40, 40, 640, 480)
    assert x0 + side <= 640 and y0 + side <= 480        # shifted back inside near the border
    assert ff.square_box(0, 0, 900, 900, 640, 480)[2] == 480  # never larger than the frame


def test_video_splits_keep_pairs_together(tmp_path, monkeypatch):
    for split, pairs in {"train": [["071", "054"]], "val": [["001", "002"]], "test": [["003", "004"]]}.items():
        (tmp_path / f"{split}.json").write_text(json.dumps(pairs))
    monkeypatch.setattr(ff, "SPLIT_DIR", tmp_path)
    s = ff.video_splits()
    assert s["071"] == s["054"] == s["071_054"] == s["054_071"] == "train"
    assert s["002_001"] == "val" and s["004"] == "test"
