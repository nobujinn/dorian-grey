import pandas as pd
from PIL import Image

from dorian_gray.experiments.external import load_manifest


def test_manifest_from_folders_and_csv(tmp_path):
    for sub, n in [("real/a", 2), ("fake", 3)]:
        (tmp_path / sub).mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (8, 8)).save(tmp_path / sub / f"{i}.png")
    (tmp_path / "fake" / "notes.txt").write_text("not an image")
    df = load_manifest(tmp_path, None)
    assert sorted(df.label.tolist()) == [0, 0, 1, 1, 1]  # recursive, non-images skipped

    pd.DataFrame({"path": ["real/a/0.png", "fake/1.png"], "label": [0, 1]}).to_csv(tmp_path / "m.csv", index=False)
    df = load_manifest(tmp_path, "m.csv")
    assert df.label.tolist() == [0, 1] and all(p.endswith(".png") for p in df.path)
