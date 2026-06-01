"""
Large grayscale ramp colour checker — 10 columns × 5 rows = 50 patches.

Patches progress from pure black (patch 0,0) to pure white (patch 4,9)
in perceptually uniform steps through non-linear sRGB.

Grid layout: 10 columns × 5 rows, left-to-right, top-to-bottom.
ArUco marker IDs 4–7 (DICT_5X5_50) so this card can coexist with
colorchecker_24 (IDs 0–3) in the same scene.
"""

import numpy as np

# ── patch names ──────────────────────────────────────────────────────────────

GRID_COLS = 10
GRID_ROWS = 5
N_PATCHES = GRID_COLS * GRID_ROWS  # 50

# sRGB code value for each patch (0–255, non-linear), evenly spaced
_VALUES_255 = np.linspace(0, 255, N_PATCHES).round().astype(np.float32)

PATCH_NAMES = [f"Gray {int(v)}" for v in _VALUES_255]

# ── reference colours ─────────────────────────────────────────────────────────

# Each patch is a neutral grey: R == G == B
_REFERENCE_SRGB_255 = np.column_stack([_VALUES_255, _VALUES_255, _VALUES_255])

# Normalised non-linear sRGB [0, 1]
REFERENCE_SRGB = (_REFERENCE_SRGB_255 / 255.0).astype(np.float32)

# ── grid geometry ─────────────────────────────────────────────────────────────

# Aspect ratio of the colour grid area (width / height)
GRID_ASPECT_RATIO = GRID_COLS / GRID_ROWS  # 2.0

# Default output size for perspective-corrected grid image
WARP_WIDTH  = 1000
WARP_HEIGHT = 500

# ── ArUco marker assignments ──────────────────────────────────────────────────

# Use IDs 4–7 to avoid collisions with colorchecker_24 (IDs 0–3).
MARKER_IDS = {
    "top_left":     4,
    "top_right":    5,
    "bottom_right": 6,
    "bottom_left":  7,
}

# Which of the 4 detected ArUco corners of each marker touches the grid.
# OpenCV ArUco corner order per marker: [TL=0, TR=1, BR=2, BL=3]
MARKER_INNER_CORNER_INDEX = {
    "top_left":     2,  # BR of marker  → grid TL
    "top_right":    3,  # BL of marker  → grid TR
    "bottom_right": 0,  # TL of marker  → grid BR
    "bottom_left":  1,  # TR of marker  → grid BL
}
