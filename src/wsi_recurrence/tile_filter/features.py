from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import sobel


def compute_features(img_pil: Image.Image) -> dict:
    """
    Canonical handcrafted features used by the saved `tile_filter_model.joblib`.

    Important: do not change feature names or computation details without
    retraining the model.
    """
    img = np.array(img_pil.convert("RGB"))
    gray = rgb2gray(img)
    hsv = rgb2hsv(img)

    gx = float(np.abs(np.diff(gray, axis=1)).mean())
    gy = float(np.abs(np.diff(gray, axis=0)).mean())

    return {
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "tissue_frac": float((gray < 0.80).mean()),
        "sat_mean": float(hsv[..., 1].mean()),
        "edge_mean": float(sobel(gray).mean()),
        "grad_x": gx,
        "grad_y": gy,
        "grad_ratio": float(max(gx, gy) / (min(gx, gy) + 1e-8)),
    }

