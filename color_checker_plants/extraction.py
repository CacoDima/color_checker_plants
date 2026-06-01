"""
Colour extraction from a perspective-corrected grid image.

Why sigma-clipped mean instead of simple mean or mode:
  - Simple mean is pulled by dust specs, reflections, and border bleed between patches.
  - Statistical mode is ill-defined for continuous colour data; KDE mode is expensive.
  - Sigma-clipped mean iteratively removes pixels that deviate more than k·σ from the
    per-channel mean, then re-averages.  It ignores specular highlights and dust while
    using all "good" pixels — more stable than median on clean patches, more robust than
    mean on noisy ones.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def extract_grid_colors(
    warped: NDArray,
    grid_cols: int,
    grid_rows: int,
    cell_margin: float = 0.15,
    method: str = "sigma_clip",
    sigma: float = 2.5,
    max_iterations: int = 5,
) -> NDArray:
    """
    Extract one representative colour per cell from a warped grid image.

    Parameters
    ----------
    warped       : (H, W, 3) float32 image, already perspective-corrected so the
                   colour grid fills the frame exactly.
    grid_cols    : Number of colour columns.
    grid_rows    : Number of colour rows.
    cell_margin  : Fraction of each cell edge to ignore (avoids inter-patch borders).
                   0.15 means the inner 70 % of each cell is sampled.
    method       : 'sigma_clip' (recommended), 'median', or 'trimmed_mean'.
    sigma        : Rejection threshold for sigma_clip (in standard deviations).
    max_iterations: Maximum sigma-clipping passes.

    Returns
    -------
    colors : (grid_rows * grid_cols, 3) float32 array, row-major (left→right, top→bottom).
    """
    H, W = warped.shape[:2]
    cell_h = H / grid_rows
    cell_w = W / grid_cols

    colors = np.empty((grid_rows * grid_cols, 3), dtype=np.float32)

    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = int(row * cell_h)
            y1 = int((row + 1) * cell_h)
            x0 = int(col * cell_w)
            x1 = int((col + 1) * cell_w)

            # Shrink by margin to avoid edge bleed
            dy = max(1, int((y1 - y0) * cell_margin))
            dx = max(1, int((x1 - x0) * cell_margin))
            patch = warped[y0 + dy : y1 - dy, x0 + dx : x1 - dx]

            pixels = patch.reshape(-1, 3).astype(np.float32)
            idx = row * grid_cols + col
            colors[idx] = _aggregate(pixels, method, sigma, max_iterations)

    return colors


# ---------------------------------------------------------------------------
# Aggregation methods
# ---------------------------------------------------------------------------

def _aggregate(
    pixels: NDArray,
    method: str,
    sigma: float,
    max_iterations: int,
) -> NDArray:
    if method == "sigma_clip":
        return _sigma_clipped_mean(pixels, sigma, max_iterations)
    elif method == "median":
        return np.median(pixels, axis=0)
    elif method == "trimmed_mean":
        return _trimmed_mean(pixels, trim=0.1)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'sigma_clip', 'median', or 'trimmed_mean'.")


def _sigma_clipped_mean(pixels: NDArray, sigma: float, max_iter: int) -> NDArray:
    mask = np.ones(len(pixels), dtype=bool)
    for _ in range(max_iter):
        if mask.sum() == 0:
            break
        subset = pixels[mask]
        mean = subset.mean(axis=0)
        std  = subset.std(axis=0)
        # Avoid division by zero on uniform patches
        std = np.where(std < 1e-6, 1e-6, std)
        distances = np.abs(pixels - mean) / std  # (N, 3)
        new_mask = (distances < sigma).all(axis=1)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    if mask.sum() == 0:
        return pixels.mean(axis=0)
    return pixels[mask].mean(axis=0)


def _trimmed_mean(pixels: NDArray, trim: float) -> NDArray:
    n = len(pixels)
    k = max(1, int(n * trim))
    result = np.empty(3, dtype=np.float32)
    for c in range(3):
        channel = np.sort(pixels[:, c])
        result[c] = channel[k : n - k].mean()
    return result
