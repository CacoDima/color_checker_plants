"""
Plant photography colour checker — 6 columns × 4 rows = 24 patches.

Designed for green-plant and sunflower scenes.  Covers the full
phenological arc (fresh leaf → chlorosis → senescence), sunflower
petals and disk, soil, and a 6-step neutral ramp for white balance.

Grid layout: 6 columns × 4 rows, left-to-right, top-to-bottom.
ArUco marker IDs 0–3 (DICT_5X5_50) — same as colorchecker_24.
"""

import numpy as np

# ── patch names ───────────────────────────────────────────────────────────────

PATCH_NAMES = [
    # Row 0 — fresh greens
    "Pale Leaf",    "Bright Leaf",  "Mature Leaf",  "Dark Leaf",
    "Deep Canopy",  "Stem Green",
    # Row 1 — stress / yellowing / senescence
    "Early Chlorosis", "Moderate Chlorosis", "Strong Yellowing", "Autumn Yellow",
    "Autumn Orange",   "Senescent Brown",
    # Row 2 — sunflower petals + disk + soil
    "Petal Bright", "Petal Mid",  "Petal Deep",   "Petal Inner",
    "Disk Centre",  "Disk Bract",
    # Row 3 — neutral ramp (WB calibration)
    "Near White",  "Light Gray",  "Mid Gray",  "Dark Gray",
    "Soil Light",  "Soil Dark",
]

# ── reference colours (non-linear sRGB, 0–255) ───────────────────────────────

_REFERENCE_SRGB_255 = np.array([
    # Row 0 — fresh greens
    [165, 200, 100],   # Pale Leaf
    [120, 165,  65],   # Bright Leaf
    [ 90, 140,  55],   # Mature Leaf
    [ 65, 110,  45],   # Dark Leaf
    [ 45,  85,  35],   # Deep Canopy
    [100, 140,  80],   # Stem Green
    # Row 1 — stress / yellowing / senescence
    [210, 220,  90],   # Early Chlorosis
    [225, 205,  65],   # Moderate Chlorosis
    [230, 185,  50],   # Strong Yellowing
    [215, 160,  45],   # Autumn Yellow
    [200, 130,  40],   # Autumn Orange
    [175, 100,  35],   # Senescent Brown
    # Row 2 — sunflower petals + disk + soil
    [255, 215,  30],   # Petal Bright
    [245, 195,  35],   # Petal Mid
    [225, 170,  40],   # Petal Deep
    [200, 145,  30],   # Petal Inner
    [ 90,  60,  25],   # Disk Centre
    [130,  90,  40],   # Disk Bract
    # Row 3 — neutral ramp
    [240, 238, 232],   # Near White
    [190, 188, 184],   # Light Gray
    [130, 128, 125],   # Mid Gray
    [ 80,  78,  76],   # Dark Gray
    [130,  95,  60],   # Soil Light
    [100,  70,  40],   # Soil Dark
], dtype=np.float32)

# Normalised non-linear sRGB [0, 1]
REFERENCE_SRGB = _REFERENCE_SRGB_255 / 255.0

# ── grid geometry ─────────────────────────────────────────────────────────────

GRID_COLS = 6
GRID_ROWS = 4
N_PATCHES = GRID_COLS * GRID_ROWS  # 24

GRID_ASPECT_RATIO = 1.5

WARP_WIDTH  = 900
WARP_HEIGHT = 600

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
