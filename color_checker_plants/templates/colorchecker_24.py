"""
X-Rite ColorChecker Classic 24-patch reference data.

Reference values are in linear sRGB (D50 illuminant), range [0, 1].
Source: colour-science library / X-Rite published specifications.

Grid layout: 6 columns × 4 rows, left-to-right, top-to-bottom.
"""

import numpy as np

# Patch names in grid order (row-major, top-left to bottom-right)
PATCH_NAMES = [
    "Dark Skin", "Light Skin", "Blue Sky", "Foliage", "Blue Flower", "Bluish Green",
    "Orange", "Purplish Blue", "Moderate Red", "Purple", "Yellow Green", "Orange Yellow",
    "Blue", "Green", "Red", "Yellow", "Magenta", "Cyan",
    "White", "Neutral 8", "Neutral 6.5", "Neutral 5", "Neutral 3.5", "Black",
]

# Reference sRGB values (non-linear, 0–255 integers) from X-Rite spec
# These are the widely-used Babel Average values
_REFERENCE_SRGB_255 = np.array([
    [115,  82,  68],
    [194, 150, 130],
    [ 98, 122, 157],
    [ 87, 108,  67],
    [133, 128, 177],
    [103, 189, 170],
    [214, 126,  44],
    [ 80,  91, 166],
    [193,  90,  99],
    [ 94,  60, 108],
    [157, 188,  64],
    [224, 163,  46],
    [ 56,  61, 150],
    [ 70, 148,  73],
    [175,  54,  60],
    [231, 199,  31],
    [187,  86, 149],
    [  8, 133, 161],
    [243, 243, 242],
    [200, 200, 200],
    [160, 160, 160],
    [122, 122, 121],
    [ 85,  85,  85],
    [ 52,  52,  52],
], dtype=np.float32)

# Normalised non-linear sRGB [0, 1]
REFERENCE_SRGB = _REFERENCE_SRGB_255 / 255.0

# Grid dimensions
GRID_COLS = 6
GRID_ROWS = 4
N_PATCHES = GRID_COLS * GRID_ROWS  # 24

# ArUco marker IDs assigned to each corner of the physical card.
# Printed card has these markers placed at the four corners with the
# colour grid inside; the inner corner of each marker touches the grid boundary.
MARKER_IDS = {
    "top_left":     0,
    "top_right":    1,
    "bottom_right": 2,
    "bottom_left":  3,
}

# For each card-corner marker, which of its 4 detected ArUco corners
# is the one touching the colour grid.
# OpenCV ArUco corner order per marker: [TL=0, TR=1, BR=2, BL=3]
MARKER_INNER_CORNER_INDEX = {
    "top_left":     2,  # BR of marker  → grid TL
    "top_right":    3,  # BL of marker  → grid TR
    "bottom_right": 0,  # TL of marker  → grid BR
    "bottom_left":  1,  # TR of marker  → grid BL
}

# Aspect ratio of the colour grid area (width / height), matches Classic card
GRID_ASPECT_RATIO = 1.5

# Default output size for perspective-corrected grid image
WARP_WIDTH  = 900
WARP_HEIGHT = 600
