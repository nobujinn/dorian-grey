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


def test_random_augment_is_deterministic_and_in_range():
    from dorian_gray.perturb import random_augment
    params = [random_augment(_image("RGB"), np.random.default_rng([1, s]))[1] for s in range(200)]
    assert params == [random_augment(_image("RGB"), np.random.default_rng([1, s]))[1] for s in range(200)]
    qualities, scales = [p["jpeg_quality"] for p in params], [p["scale"] for p in params]
    assert all(q == 0 or 30 <= q <= 100 for q in qualities) and all(0.5 <= s <= 1.5 for s in scales)
    assert 0.35 < np.mean([q > 0 for q in qualities]) < 0.65 and 0.35 < np.mean([s != 1 for s in scales]) < 0.65


def test_deg_augment_adds_blur_and_noise():
    from dorian_gray.perturb import random_augment
    params = [random_augment(_image("RGB"), np.random.default_rng([1, s]), "deg")[1] for s in range(300)]
    blur, noise = np.array([p["blur_sigma"] for p in params]), np.array([p["noise_sigma"] for p in params])
    assert ((blur == 0) | ((blur >= 0.5) & (blur <= 2))).all() and ((noise == 0) | ((noise >= 1) & (noise <= 6))).all()
    assert 0.2 < (blur > 0).mean() < 0.4 and 0.2 < (noise > 0).mean() < 0.4
    # the original scheme draws exactly as before: deg does not change rj caches
    rj = [random_augment(_image("RGB"), np.random.default_rng([1, s]))[1] for s in range(20)]
    assert all(set(p) == {"jpeg_quality", "scale"} for p in rj)


def test_unseen_operations():
    im = _image("RGB")
    assert apply(im, "webp75").format == "WEBP" and apply(im, "webp75").size == im.size
    blurred = np.asarray(apply(im, "blur1"), float)
    assert np.abs(np.diff(blurred, axis=1)).mean() < np.abs(np.diff(np.asarray(im, float), axis=1)).mean()
    noisy = np.asarray(apply(im, "noise3"), float) - np.asarray(im, float)
    assert 1.5 < noisy.std() < 4 and np.array_equal(apply(im, "noise3"), apply(im, "noise3"))  # seeded


def test_social_chains_resize_and_jpeg():
    out = apply(_image("RGB"), "social")
    assert out.size == (72, 48) and out.format == "JPEG"
