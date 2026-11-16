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
