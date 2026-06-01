"""
Main detection pipeline: ArUco → perspective warp → colour extraction.

Typical usage:

    from color_checker_plants import detect_color_checker, fit_correction, apply_correction

    result = detect_color_checker(image)
    if result is not None:
        model = fit_correction(result.measured_colors, result.reference_colors)
        corrected = apply_correction(image, model)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .markers import detect_markers, extract_grid_corners, draw_markers
from .extraction import extract_grid_colors
from .templates.colorchecker_24 import (
    REFERENCE_SRGB,
    GRID_COLS,
    GRID_ROWS,
    MARKER_IDS,
    WARP_WIDTH,
    WARP_HEIGHT,
)


@dataclass
class DetectionResult:
    """All outputs from a successful colour checker detection."""

    # Colours extracted from the photo, shape (N, 3), float32 [0, 1]
    measured_colors: NDArray

    # Reference ground-truth colours for the same patches, shape (N, 3)
    reference_colors: NDArray

    # Perspective-corrected crop of just the colour grid, (H, W, 3) float32
    warped_image: NDArray

    # The 4 grid corner points found in the original image, shape (4, 2) float32
    grid_corners: NDArray

    # Original image with marker overlays drawn, for debugging
    debug_image: NDArray


def detect_color_checker(
    image: NDArray,
    grid_cols: int = GRID_COLS,
    grid_rows: int = GRID_ROWS,
    reference_colors: NDArray | None = None,
    marker_ids: dict[str, int] | None = None,
    warp_width: int = WARP_WIDTH,
    warp_height: int = WARP_HEIGHT,
    cell_margin: float = 0.15,
    extraction_method: str = "sigma_clip",
    sigma: float = 2.5,
) -> DetectionResult | None:
    """
    Detect a marker-framed colour checker in an image and extract patch colours.

    Parameters
    ----------
    image            : (H, W, 3) or (H, W, 4) uint8 or float32 image.
                       If float, values must be in [0, 1].
    grid_cols        : Number of columns in the colour grid.
    grid_rows        : Number of rows in the colour grid.
    reference_colors : (N, 3) reference sRGB values for the patches.
                       Defaults to the 24-patch classic reference.
    marker_ids       : Mapping of corner name → ArUco ID.
    warp_width/height: Size of the perspective-corrected output.
    cell_margin      : Fraction of each cell border to discard before sampling.
    extraction_method: 'sigma_clip', 'median', or 'trimmed_mean'.
    sigma            : Sigma-clipping threshold (used when method='sigma_clip').

    Returns
    -------
    DetectionResult, or None if the four corner markers were not found.
    """
    if reference_colors is None:
        reference_colors = REFERENCE_SRGB

    img_float = _to_float32(image)
    img_bgr   = _to_bgr_uint8(image)

    corners, ids = detect_markers(img_bgr)
    grid_corners = extract_grid_corners(corners, ids, marker_ids)

    if grid_corners is None:
        return None

    warped  = _warp_grid(img_float, grid_corners, warp_width, warp_height)
    colors  = extract_grid_colors(
        warped, grid_cols, grid_rows, cell_margin, extraction_method, sigma
    )
    debug   = draw_markers(img_bgr, corners, ids)

    return DetectionResult(
        measured_colors  = colors,
        reference_colors = reference_colors.astype(np.float32),
        warped_image     = warped,
        grid_corners     = grid_corners,
        debug_image      = debug,
    )


# ---------------------------------------------------------------------------
# Perspective warp
# ---------------------------------------------------------------------------

def _warp_grid(
    image: NDArray,
    grid_corners: NDArray,
    out_w: int,
    out_h: int,
) -> NDArray:
    """
    Warp the image so that grid_corners map to the four corners of (out_w × out_h).
    grid_corners order: [TL, TR, BR, BL].
    """
    dst = np.array([
        [0,         0        ],
        [out_w - 1, 0        ],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(grid_corners, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return warped.astype(np.float32)


# ---------------------------------------------------------------------------
# Image normalisation helpers
# ---------------------------------------------------------------------------

def _to_float32(image: NDArray) -> NDArray:
    img = np.asarray(image)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def _to_bgr_uint8(image: NDArray) -> NDArray:
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 3:
        # Assume RGB input → convert to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
