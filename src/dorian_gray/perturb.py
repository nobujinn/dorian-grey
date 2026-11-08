"""Post-processing applied to test images for the robustness evaluation.

These mimic what happens to an image shared online (`corvi2023detection`, `grommelt2024fake`) and are
applied to real and fake test images alike. JPEG and resizing are also the training-time augmentation
(below); blur, noise and WebP are never seen in training with the original scheme (`rj`), so they test
robustness to *unknown* operations (the `deg` scheme adds blur and noise to training). `social` chains a
downscale and JPEG, as a social network does on upload.
"""

import io
import zlib

import numpy as np
from PIL import Image, ImageFilter

# name -> chain of (operation, parameter) steps; "clean" is the unmodified image
CONDITIONS = {
    "clean": [],
    "jpeg95": [("jpeg", 95)], "jpeg85": [("jpeg", 85)], "jpeg75": [("jpeg", 75)], "jpeg50": [("jpeg", 50)],
    "resize0.5": [("resize", 0.5)], "resize0.75": [("resize", 0.75)], "resize1.5": [("resize", 1.5)],
    # not used in training
    "blur1": [("blur", 1.0)], "noise3": [("noise", 3.0)], "webp75": [("webp", 75)],
    "social": [("resize", 0.75), ("jpeg", 80)],
}


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode with libjpeg via Pillow (default 4:2:0 chroma subsampling, as most web encoders)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


def webp(image: Image.Image, quality: int) -> Image.Image:
    """Lossy WebP (libwebp via Pillow): block transform coding like JPEG, but 4x4 blocks with intra
    prediction and an in-loop deblocking filter, so it leaves no 8x8 grid."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="WEBP", quality=quality)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


def blur(image: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur: removes exactly the high frequencies most detectors rely on."""
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(sigma))


def noise(image: Image.Image, sigma: float) -> Image.Image:
    """Additive Gaussian noise (std in 0-255 units), e.g. sensor noise of a re-capture. The noise is
    seeded by the image content, so it is reproducible but differs between images."""
    x = np.asarray(image.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(zlib.crc32(x.tobytes()))
    return Image.fromarray(np.clip(np.rint(x + rng.normal(0, sigma, x.shape)), 0, 255).astype(np.uint8))


def resize(image: Image.Image, scale: float) -> Image.Image:
    """Bicubic rescale; Pillow widens the filter support when downscaling, so it is antialiased."""
    w, h = image.size
    return image.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)


OPERATIONS = {"jpeg": jpeg, "webp": webp, "resize": resize, "blur": blur, "noise": noise}


def apply(image: Image.Image, condition: str) -> Image.Image:
    for op, param in CONDITIONS[condition]:
        image = OPERATIONS[op](image, param)
    return image


# training-time augmentation. Two schemes, chosen per run (cache suffix `_aug` / `_aug-deg`):
#   rj  (`wang2020cnn` used random JPEG + blur; here JPEG + resize, the two operations the robustness test
#       measures). Resize first, then JPEG, as in a typical "edit, then save" pipeline.
#   deg rj preceded by random blur and noise (results log 16: the remaining weakness is distorted images, e.g.
#       NTIRE). Blur and noise are then no longer unseen operations for these models; WebP still is.
AUG_P, AUG_SCALE, AUG_QUALITY = 0.5, (0.5, 1.5), (30, 100)
AUG_DEG_P, AUG_BLUR, AUG_NOISE = 0.3, (0.5, 2.0), (1.0, 6.0)
AUG_SCHEMES = ["rj", "deg"]


def aug_suffix(scheme: str) -> str:
    """Cache / run-name suffix of an augmentation scheme (`rj` keeps the original name `_aug`)."""
    return "_aug" if scheme == "rj" else f"_aug-{scheme}"


def random_augment(image: Image.Image, rng, scheme: str = "rj") -> tuple[Image.Image, dict]:
    """The augmented image and its parameters: jpeg_quality (0 = not recompressed), scale (1 = not resized),
    for `deg` also blur_sigma and noise_sigma (0 = not applied). The `rj` draws are unchanged from the
    first version, so its cached copies stay reproducible."""
    params = {}
    if scheme == "deg":
        params["blur_sigma"] = float(rng.uniform(*AUG_BLUR)) if rng.random() < AUG_DEG_P else 0.0
        params["noise_sigma"] = float(rng.uniform(*AUG_NOISE)) if rng.random() < AUG_DEG_P else 0.0
        if params["blur_sigma"]:
            image = blur(image, params["blur_sigma"])
        if params["noise_sigma"]:
            image = noise(image, params["noise_sigma"])
    elif scheme != "rj":
        raise ValueError(f"unknown augmentation scheme {scheme}")
    quality, scale = 0, 1.0
    if rng.random() < AUG_P:
        scale = float(rng.uniform(*AUG_SCALE))
        image = resize(image, scale)
    if rng.random() < AUG_P:
        quality = int(rng.integers(AUG_QUALITY[0], AUG_QUALITY[1] + 1))
        image = jpeg(image, quality)
    return image, {"jpeg_quality": quality, "scale": scale, **params}
