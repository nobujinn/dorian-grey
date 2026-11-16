"""Aligned fake training images: real photos passed through a latent-diffusion autoencoder (encode + decode).

Idea (AlignedForensics, Rajan et al., ICLR 2025; also B-Free, DDA): a reconstruction keeps the content, size and
file history of its real original and adds only the generator's decoder traces. Training "real vs. its own
reconstruction" leaves no content or format shortcut, so the detector has to learn the decoder fingerprint, which
every latent-diffusion generator shares. Limit: it teaches latent-diffusion traces, not GAN or pixel-space ones.

Autoencoder: `stabilityai/sd-vae-ft-mse` (the Stable Diffusion 1.x VAE with the fine-tuned decoder), fp16 on GPU.
Images larger than MAX_SIDE are center-cropped to it first; the crop offset and size are multiples of 8, so the 8x8
JPEG grid of the original is kept. 320 px covers what both methods look at (Method A: the central <= 256 px crop,
Method B: the central 224 px crop) with a 32 px margin against the autoencoder's border effects, and keeps the cost
at ~0.8 s per image on the GTX 1050 (512 px: 1.8 s, 1 GB VRAM; 768 px: 10 s, 2 GB). The latent mean is decoded
(no sampling noise), so the reconstruction is deterministic.
"""

import numpy as np
import torch
from PIL import Image

from dorian_gray.config import device

VAE_NAME = "stabilityai/sd-vae-ft-mse"
MAX_SIDE = 320


def center_crop8(image: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Central crop with both sides <= max_side and multiples of 8, offsets multiples of 8."""
    w, h = image.size
    cw, ch = min(w, max_side) // 8 * 8, min(h, max_side) // 8 * 8
    left, top = (w - cw) // 2 // 8 * 8, (h - ch) // 2 // 8 * 8
    return image.crop((left, top, left + cw, top + ch))


class VaeReconstructor:
    def __init__(self, name: str = VAE_NAME):
        from diffusers import AutoencoderKL
        self.device = device()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.vae = AutoencoderKL.from_pretrained(name, torch_dtype=self.dtype).to(self.device).eval()

    @torch.inference_mode()
    def _roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(self.vae.encode(x).latent_dist.mean).sample

    def __call__(self, image: Image.Image) -> Image.Image:
        im = center_crop8(image.convert("RGB"))
        x = torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        y = self._roundtrip(x.to(self.device, self.dtype))
        if not torch.isfinite(y).all():  # fp16 overflow: redo this image in fp32
            self.vae.float()
            y = self._roundtrip(x.to(self.device, torch.float32))
            self.vae.to(self.dtype)
        y = ((y[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(y, "RGB")
