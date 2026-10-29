import numpy as np
from PIL import Image

from dorian_gray.perturb import CONDITIONS, apply


def _image(mode="RGBA"):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (64, 96, 4), dtype=np.uint8), "RGBA").convert(mode)


def test_clean_is_identity():
    im = _image()
    assert apply(im, "clean") is im


def test_resize_scales_both_sides():
    assert apply(_image(), "resize0.5").size == (48, 32)
    assert apply(_image(), "resize1.5").size == (144, 96)


def test_jpeg_is_lossy_and_keeps_size():
    im = _image("RGB")
    out = apply(im, "jpeg50")
    assert out.format == "JPEG" and out.size == im.size
    assert np.abs(np.asarray(out, float) - np.asarray(im, float)).mean() > 1
    # lower quality = larger error
    err = lambda c: np.abs(np.asarray(apply(im, c), float) - np.asarray(im, float)).mean()
    assert err("jpeg95") < err("jpeg75") < err("jpeg50")


def test_every_condition_returns_rgb_or_original():
    for c in CONDITIONS:
        assert apply(_image("P"), c).mode in {"RGB", "P"}
