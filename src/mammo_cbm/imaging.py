"""Image loading and preprocessing. The same functions are used for training and for the dashboard."""

import numpy as np
import torch
from PIL import Image

IMG = 224
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_array(src):
    """Path or file-like -> float32 2-D array, bright = dense tissue. Handles DICOM (incl. MONOCHROME1), PNG, JPG."""
    name = str(getattr(src, "name", src)).lower()
    if name.endswith((".dcm", ".dicom")):
        import pydicom

        ds = pydicom.dcmread(src)
        a = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            a = a.max() - a
        return a
    im = Image.open(src)
    if im.mode.startswith("I") or im.mode == "F":  # 16-bit: keep full range
        return np.asarray(im, dtype=np.float32)
    return np.asarray(im.convert("L"), dtype=np.float32)


def roi_crop(a, box, margin=0.25):
    """Square crop around box=(y0, x0, y1, x1) with context margin, padded at the image edge."""
    y0, x0, y1, x1 = (int(v) for v in box)
    s = max(32, int(max(y1 - y0, x1 - x0) * (1 + 2 * margin)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    ys, xs = max(0, cy - s // 2), max(0, cx - s // 2)
    c = a[ys : ys + s, xs : xs + s]
    return np.pad(c, ((0, s - c.shape[0]), (0, s - c.shape[1])), constant_values=float(c.min()))


def prep_gray(crop):
    """Percentile contrast stretch + resize -> float32 IMGxIMG in [0, 1]."""
    lo, hi = np.percentile(crop, [0.5, 99.5])
    c = np.clip((crop - lo) / (hi - lo + 1e-6), 0, 1)
    im = Image.fromarray((c * 255).astype(np.uint8)).resize((IMG, IMG), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def normalise(g):
    """(H, W) array or tensor in [0, 1] -> normalised 3xHxW tensor."""
    g = torch.as_tensor(g, dtype=torch.float32)
    return (g[None].expand(3, -1, -1) - _MEAN) / _STD
