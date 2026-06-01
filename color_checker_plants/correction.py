"""
Apply a fitted colour correction model to full images.

The model is computed per-patch on the colour checker and then applied
pixel-wise to the entire photograph.  The correction is purely RGB — it does
not require any colour space conversion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .fitting import apply_correction_to_colors


def apply_correction(image: NDArray, model: dict) -> NDArray:
    """
    Apply a colour correction model to an image.

    Parameters
    ----------
    image : (H, W, 3) float array, values in [0, 1].
    model : dict returned by fitting.fit_correction().

    Returns
    -------
    corrected : (H, W, 3) float32, clipped to [0, 1].
    """
    img = np.asarray(image, dtype=np.float32)
    H, W, _ = img.shape
    pixels = img.reshape(-1, 3)
    corrected_pixels = apply_correction_to_colors(pixels, model)
    return corrected_pixels.reshape(H, W, 3)


def correction_gain_map(image: NDArray, model: dict) -> NDArray:
    """
    Return a (H, W, 3) gain map showing how much each pixel was shifted.

    Useful for visualising where the correction has the largest effect.
    Values > 1 mean the channel was boosted; < 1 means it was reduced.
    """
    corrected = apply_correction(image, model)
    img = np.asarray(image, dtype=np.float32)
    eps = 1e-6
    return np.clip(corrected / (img + eps), 0.0, 4.0)


def white_balance_from_model(model: dict) -> NDArray:
    """
    Extract approximate per-channel white-balance multipliers from the model.

    Uses the neutral-8 patch (index 19 in the 24-patch checker) diagonal of
    the correction matrix as a proxy for WB gains.  Works only for 'matrix' method.

    Returns (3,) array of [R_gain, G_gain, B_gain].
    """
    if model["method"] != "matrix":
        raise ValueError("white_balance_from_model() only supports method='matrix'.")
    W = model["matrix"][:3, :3]  # drop bias row
    return np.diag(W).astype(np.float32)
