"""Interpretable concept-bottleneck mammography lesion classifier (research prototype)."""

from .conformal import conformal_thresholds, verdict_from_p
from .imaging import IMG, load_array, normalise, prep_gray, roi_crop
from .inference import Bundle, analyse, load_bundle, save_bundle
from .model import CBM, gradcam

__version__ = "0.1.0"
__all__ = [
    "CBM",
    "IMG",
    "Bundle",
    "analyse",
    "conformal_thresholds",
    "gradcam",
    "load_array",
    "load_bundle",
    "normalise",
    "prep_gray",
    "roi_crop",
    "save_bundle",
    "verdict_from_p",
]
