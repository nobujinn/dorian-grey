"""Method B: frozen CLIP ViT-L/14 image embeddings + logistic-regression probe (UnivFD-style).

Ojha et al., CVPR 2023 (`ojha2023towards`). The backbone is never trained, so every image is
embedded once and the 768-d embeddings are cached; only the linear probe is fitted.
"""

from pathlib import Path

import joblib
import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from torchvision.transforms import v2 as T

from dorian_gray.config import device

MODEL_NAME, PRETRAINED = "ViT-L-14", "openai"
INPUT_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
_crop_norm = T.Compose([T.CenterCrop(INPUT_SIZE), T.Normalize(CLIP_MEAN, CLIP_STD)])


def preprocess(image: Image.Image) -> torch.Tensor:
    """Center-crop instead of resizing: downscaling smooths away generator traces
    (UnivFD test protocol; `li2025safe`). Only images smaller than 224 px are upscaled."""
    x = _to_tensor(image.convert("RGB"))
    if min(x.shape[-2:]) < INPUT_SIZE:
        x = T.functional.resize(x, INPUT_SIZE, interpolation=T.InterpolationMode.BICUBIC, antialias=True)
    return _crop_norm(x)


class ClipEmbedder:
    """Only the visual tower, in fp16 on GPU (fits a 3 GB GTX 1050) or fp32 on CPU."""

    def __init__(self):
        self.device = device()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        self.visual = model.visual.to(self.device, self.dtype).eval()
        del model

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([preprocess(im) for im in images]).to(self.device, self.dtype)
        return self.visual(x).float().cpu().numpy()


def make_probe(C: float = 1.0) -> "Pipeline":
    """L2-normalise embeddings (as CLIP does), then logistic regression.

    Logistic regression already outputs a sigmoid of a linear score, i.e. a Platt-shaped
    probability; it still has to be checked for calibration on held-out data.
    """
    return make_pipeline(Normalizer(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def save_probe(probe, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(probe, path)


def load_probe(path: Path):
    return joblib.load(path)
