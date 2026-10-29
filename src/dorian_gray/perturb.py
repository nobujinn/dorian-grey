"""Post-processing applied to test images for the robustness evaluation (JPEG recompression, resizing).

These mimic what happens to an image shared online (`corvi2023detection`, `grommelt2024fake`). Both
are applied to real and fake test images alike; detectors are always trained on clean images.
"""

import io

from PIL import Image

# name -> (operation, parameter); "clean" is the unmodified image
CONDITIONS = {
    "clean": None,
    "jpeg95": ("jpeg", 95), "jpeg85": ("jpeg", 85), "jpeg75": ("jpeg", 75), "jpeg50": ("jpeg", 50),
    "resize0.5": ("resize", 0.5), "resize0.75": ("resize", 0.75), "resize1.5": ("resize", 1.5),
}


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode with libjpeg via Pillow (default 4:2:0 chroma subsampling, as most web encoders)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


def resize(image: Image.Image, scale: float) -> Image.Image:
    """Bicubic rescale; Pillow widens the filter support when downscaling, so it is antialiased."""
    w, h = image.size
    return image.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)


def apply(image: Image.Image, condition: str) -> Image.Image:
    spec = CONDITIONS[condition]
    if spec is None:
        return image
    op, param = spec
    return jpeg(image, param) if op == "jpeg" else resize(image, param)
