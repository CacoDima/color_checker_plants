"""Miscellaneous utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def delta_e_simple(a: NDArray, b: NDArray) -> NDArray:
    """
    Approximate per-patch ΔE in sRGB space (Euclidean, not CIE ΔE2000).
    Good enough for comparing correction residuals.  Values < 0.01 are excellent.
    """
    return np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum(axis=-1))


def srgb_to_linear(srgb: NDArray) -> NDArray:
    """Apply the sRGB inverse EOTF (gamma ≈ 2.2 with linear segment)."""
    srgb = np.asarray(srgb, dtype=np.float32)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    return linear.astype(np.float32)


def linear_to_srgb(linear: NDArray) -> NDArray:
    """Apply the sRGB EOTF (encode linear light to sRGB)."""
    linear = np.asarray(linear, dtype=np.float32)
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * (linear ** (1.0 / 2.4)) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


def save_image(path: str, image: NDArray) -> None:
    """Save a float32 [0,1] or uint8 image to disk (PNG/JPEG inferred from path)."""
    import cv2
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def load_image(path: str) -> NDArray:
    """Load an image as float32 RGB [0, 1]."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
