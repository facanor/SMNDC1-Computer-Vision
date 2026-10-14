from __future__ import annotations
from pathlib import Path
import numpy as np

def imread_gray(path: str) -> np.ndarray:
    """
    Reads a grayscale image from .tif/.tiff/.png/.jpg.
    Returns float32 array (no normalization) with shape (H,W).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    # Try tifffile first (best for .tif)
    try:
        import tifffile
        img = tifffile.imread(str(p))
    except Exception:
        # fallback: imageio
        import imageio.v3 as iio
        img = iio.imread(str(p))

    img = np.asarray(img)
    if img.ndim == 3:
        # If accidentally multi-channel, take first channel
        img = img[..., 0]
    return img.astype(np.float32)
