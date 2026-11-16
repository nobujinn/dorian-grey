from PIL import Image

from dorian_gray.recon import center_crop8


def test_center_crop_keeps_8x8_grid_and_limits_size():
    for w, h in [(1000, 700), (300, 517), (512, 512), (37, 1000)]:
        out = center_crop8(Image.new("RGB", (w, h)), 512)
        assert out.size[0] % 8 == 0 and out.size[1] % 8 == 0 and max(out.size) <= 512
    # offsets are multiples of 8: a 512 crop of 1000x700 starts at (244 // 8 * 8, 94 // 8 * 8) = (240, 88)
    im = Image.new("RGB", (1000, 700))
    im.putpixel((240, 88), (255, 0, 0))
    assert center_crop8(im, 512).getpixel((0, 0)) == (255, 0, 0)


def _encode(im, fmt, **kw):
    import io
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return Image.open(io.BytesIO(buf.getvalue()))


def test_match_jpeg_copies_quantisation_and_subsampling():
    import numpy as np
    from PIL import JpegImagePlugin

    from dorian_gray.recon import match_jpeg
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")
    recon = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), "RGB")
    for quality, subsampling in [(90, 0), (60, 2)]:
        original = _encode(noise, "JPEG", quality=quality, subsampling=subsampling)
        out = match_jpeg(original, recon)
        assert out.format == "JPEG" and out.quantization == original.quantization
        assert JpegImagePlugin.get_sampling(out) == JpegImagePlugin.get_sampling(original) == subsampling
    gray = match_jpeg(_encode(noise.convert("L"), "JPEG", quality=80), recon)
    assert gray.mode == "L"
    assert match_jpeg(_encode(noise, "PNG"), recon) is recon  # non-JPEG originals: unchanged
