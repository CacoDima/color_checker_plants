"""
Sunflower colour checker — 6 columns × 2 rows = 12 patches.

Compact card for sunflower field sessions.
Row 0: petal gradient (bright → deep), disk centre, leaf, stem.
Row 1: neutral ramp (near-white → near-black + soil) for white balance.

Grid layout: 6 columns × 2 rows, left-to-right, top-to-bottom.
ArUco marker IDs 0–3 (DICT_5X5_50) — same as colorchecker_24.
"""

import numpy as np

# ── patch names ───────────────────────────────────────────────────────────────

PATCH_NAMES = [
    # Row 0 — sunflower colours
    "Petal Bright", "Petal Mid", "Petal Deep",
    "Disk Centre",  "Leaf",      "Stem",
    # Row 1 — neutral ramp
    "Near White", "Light Gray", "Mid Gray",
    "Dark Gray",  "Soil",       "Near Black",
]

# ── reference colours (non-linear sRGB, 0–255) ───────────────────────────────

_REFERENCE_SRGB_255 = np.array([
    # Row 0 — sunflower colours
    [255, 215,  30],   # Petal Bright
    [240, 185,  35],   # Petal Mid
    [220, 155,  35],   # Petal Deep
    [ 85,  55,  22],   # Disk Centre
    [110, 155,  60],   # Leaf
    [ 90, 130,  55],   # Stem
    # Row 1 — neutral ramp
    [240, 238, 232],   # Near White
    [185, 183, 180],   # Light Gray
    [125, 123, 120],   # Mid Gray
    [ 70,  68,  66],   # Dark Gray
    [125,  90,  55],   # Soil
    [ 28,  28,  28],   # Near Black
], dtype=np.float32)

# Normalised non-linear sRGB [0, 1]
REFERENCE_SRGB = _REFERENCE_SRGB_255 / 255.0

# ── grid geometry ─────────────────────────────────────────────────────────────

GRID_COLS = 6
GRID_ROWS = 2
N_PATCHES = GRID_COLS * GRID_ROWS  # 12

GRID_ASPECT_RATIO = 3.0

WARP_WIDTH  = 900
WARP_HEIGHT = 300

# ── ArUco marker assignments ──────────────────────────────────────────────────

MARKER_IDS = {
    "top_left":     0,
    "top_right":    1,
    "bottom_right": 2,
    "bottom_left":  3,
}

MARKER_INNER_CORNER_INDEX = {
    "top_left":     2,  # BR of marker  → grid TL
    "top_right":    3,  # BL of marker  → grid TR
    "bottom_right": 0,  # TL of marker  → grid BR
    "bottom_left":  1,  # TR of marker  → grid BL
}
