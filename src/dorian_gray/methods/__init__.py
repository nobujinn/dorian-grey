"""The two detection methods share one interface: an extractor (images -> feature matrix, cached once)
and a small classifier trained on the cached features."""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Method:
    features: str                 # cache name under cache/features/
    extractor: Callable[[], object]  # factory, so heavy models are only loaded when needed
    classifier: Callable[[float], object]
    cache_dtype: str = "float16"  # enough for CLIP embeddings; hand-crafted features need float32


def _clip():
    from dorian_gray.methods.clip_probe import ClipEmbedder
    return ClipEmbedder()


def _forensic():
    from dorian_gray.methods.forensic import ForensicFeatures
    return ForensicFeatures()


def _clip_probe(C):
    from dorian_gray.methods.clip_probe import make_probe
    return make_probe(C)


def _forensic_lr(C):
    from dorian_gray.methods.forensic import make_classifier
    return make_classifier(C)


METHODS = {
    "a": Method("forensic_v1", _forensic, _forensic_lr, "float32"),
    "b": Method("clip_vitl14", _clip, _clip_probe),
}
