"""
Named colour presets for generate_card.py.

Each preset is a (colors, cols, description) tuple:
  colors : (N, 3) float32 sRGB array, values in [0, 1]
  cols   : suggested number of columns for the grid
  desc   : one-line description shown in --list-presets
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .colorchecker_24 import REFERENCE_SRGB, MARKER_IDS as _DEFAULT_MARKER_IDS
from .grayscale_ramp_50 import (
    REFERENCE_SRGB as _GRAY50_SRGB,
    GRID_COLS as _GRAY50_COLS,
    MARKER_IDS as _GRAY50_MARKER_IDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb(r: int, g: int, b: int) -> list[float]:
    return [r / 255.0, g / 255.0, b / 255.0]


def _gray_ramp(n: int, lo: int = 0, hi: int = 255) -> NDArray:
    """n evenly-spaced neutral patches from hi down to lo (sRGB)."""
    vals = np.linspace(hi, lo, n, dtype=np.float32) / 255.0
    return np.stack([vals, vals, vals], axis=1)


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, tuple[NDArray, int, str]] = {}


def _reg(name: str, colors: NDArray, cols: int, desc: str):
    PRESETS[name] = (np.asarray(colors, dtype=np.float32), cols, desc)


# ── Standard X-Rite 24-patch ──────────────────────────────────────────────
_reg(
    "classic",
    REFERENCE_SRGB,
    6,
    "X-Rite ColorChecker Classic 24 patches (default)",
)

# ── Large 50-patch grayscale ramp (matches grayscale_ramp_50 template) ───────
_reg(
    "gray_50",
    _GRAY50_SRGB,
    _GRAY50_COLS,
    "50 neutral patches black->white, 10x5 grid (ArUco IDs 4-7)",
)

# ── Full neutral ramp ─────────────────────────────────────────────────────
_reg(
    "gray_24",
    _gray_ramp(24),
    6,
    "24 neutral patches, white -> black in equal sRGB steps (6x4)",
)
_reg(
    "gray_12",
    _gray_ramp(12),
    6,
    "12 neutral patches, white -> black (6x2)",
)
_reg(
    "gray_10",
    _gray_ramp(10),
    5,
    "10 neutral patches 100%->0% density, standard densitometry ramp (5x2)",
)

# ── Highlight / white gradation ───────────────────────────────────────────
# Biased toward highlights: 12 steps from pure white (255) to mid-bright (128).
# Useful for testing printer highlight reproduction and camera clipping.
_reg(
    "white_grad",
    _gray_ramp(12, lo=128, hi=255),
    6,
    "12 near-white neutral patches 100%->50% (highlight gradation, 6x2)",
)

# Wider white zone: 18 steps from 255 down to 64 (shadow edge)
_reg(
    "white_grad_wide",
    _gray_ramp(18, lo=64, hi=255),
    6,
    "18 neutral patches 100%->25%, covers highlights + upper mids (6x3)",
)

# ── Chromatic white-balance test ─────────────────────────────────────────
# 12 near-white patches with varying colour temperature bias.
# Row 1: warm (orange) tints, Row 2: cool (blue) tints.
# Useful for testing WB accuracy in different scene illuminants.
_warm_cool = np.array([
    # warm tints (R > B), descending brightness
    _rgb(255, 248, 220),  # 6500K-ish candlelight tint
    _rgb(255, 244, 214),
    _rgb(255, 240, 210),
    _rgb(252, 237, 206),
    _rgb(248, 233, 200),
    _rgb(244, 228, 194),
    # cool tints (B > R)
    _rgb(214, 228, 244),  # 8000K-ish overcast tint
    _rgb(210, 224, 248),
    _rgb(206, 220, 252),
    _rgb(200, 216, 255),
    _rgb(194, 210, 255),
    _rgb(188, 205, 255),
], dtype=np.float32)
_reg("warm_cool_white", _warm_cool, 6, "12 near-white patches: warm tints (row 1) + cool tints (row 2)")

# ── Primary/secondary colour set ─────────────────────────────────────────
_primaries = np.array([
    _rgb(220,  50,  50),  # red
    _rgb( 50, 200,  50),  # green
    _rgb( 50,  50, 220),  # blue
    _rgb( 50, 210, 210),  # cyan
    _rgb(210,  50, 210),  # magenta
    _rgb(220, 200,  50),  # yellow
    _rgb(255, 255, 255),  # white
    _rgb(  0,   0,   0),  # black
    _rgb(200, 100,  50),  # orange
    _rgb( 50, 150,  50),  # dark green
    _rgb(100,  50, 150),  # purple
    _rgb(128, 128, 128),  # mid gray
], dtype=np.float32)
_reg("primaries_12", _primaries, 6, "12 patches: primaries, secondaries, white, black + extras (6x2)")

# ── Skin-tone focused (useful for portraits / plant tissue colours) ───────
_skin = np.array([
    _rgb(245, 215, 185),  # very light
    _rgb(240, 200, 165),
    _rgb(230, 185, 145),
    _rgb(215, 165, 125),
    _rgb(200, 150, 110),
    _rgb(185, 135,  95),
    _rgb(170, 118,  80),
    _rgb(155, 100,  65),
    _rgb(140,  85,  50),
    _rgb(125,  72,  40),
    _rgb(110,  60,  32),
    _rgb( 90,  46,  22),
], dtype=np.float32)
_reg("skin_tones", _skin, 6, "12 skin-tone / warm-organic patches, light to dark (6x2)")

# ── Plant-focused 24-patch checker ───────────────────────────────────────
# Row 1: fresh greens (pale → dark canopy)
# Row 2: stressed / yellowing / autumn transition
# Row 3: sunflower petals + disk + soil/bark
# Row 4: neutral ramp for WB calibration (essential even in plant scenes)
_plants_24 = np.array([
    # Row 1 — fresh greens
    _rgb(165, 200, 100),  # 0  pale/young leaf tissue
    _rgb(120, 165,  65),  # 1  bright leaf (grass, spinach)
    _rgb( 90, 140,  55),  # 2  typical mature leaf
    _rgb( 65, 110,  45),  # 3  darker leaf (shade plant)
    _rgb( 45,  85,  35),  # 4  deep canopy / heavy shadow
    _rgb(100, 140,  80),  # 5  stem / petiole green
    # Row 2 — stress / yellowing / autumn
    _rgb(210, 220,  90),  # 6  early chlorosis (slight yellowing)
    _rgb(225, 205,  65),  # 7  moderate chlorosis
    _rgb(230, 185,  50),  # 8  strong yellowing / autumn onset
    _rgb(215, 160,  45),  # 9  autumn yellow-orange
    _rgb(200, 130,  40),  # 10 autumn orange leaf
    _rgb(175, 100,  35),  # 11 senescent / brown-orange leaf
    # Row 3 — sunflower petals + disk + soil
    _rgb(255, 215,  30),  # 12 bright sunflower petal
    _rgb(245, 195,  35),  # 13 mid petal
    _rgb(225, 170,  40),  # 14 deep / shadowed petal
    _rgb(200, 145,  30),  # 15 inner petal ring
    _rgb( 90,  60,  25),  # 16 sunflower disk (centre)
    _rgb(130,  90,  40),  # 17 disk periphery / bract
    # Row 4 — neutrals for white balance
    _rgb(240, 238, 232),  # 18 near-white
    _rgb(190, 188, 184),  # 19 light gray
    _rgb(130, 128, 125),  # 20 mid gray
    _rgb( 80,  78,  76),  # 21 dark gray
    _rgb(130,  95,  60),  # 22 soil / earth
    _rgb(100,  70,  40),  # 23 dark soil / bark
], dtype=np.float32)
_reg(
    "plants_24",
    _plants_24, 6,
    "24 patches for plant photography: greens, chlorosis, sunflower petals+disk, soil, neutrals (6x4)",
)

# ── Sunflower-focused 12-patch ────────────────────────────────────────────
# Compact card for rapid sunflower field sessions.
# Row 1: petals (bright → deep) + disk centre + leaf + stem
# Row 2: neutral ramp (WB anchors)
_sunflower_12 = np.array([
    # Row 1 — sunflower colours
    _rgb(255, 215,  30),  # 0  bright petal
    _rgb(240, 185,  35),  # 1  mid / inner petal
    _rgb(220, 155,  35),  # 2  deep / base of petal
    _rgb( 85,  55,  22),  # 3  disk centre (dark brown)
    _rgb(110, 155,  60),  # 4  leaf / bracts
    _rgb( 90, 130,  55),  # 5  stem green
    # Row 2 — neutral ramp
    _rgb(240, 238, 232),  # 6  near-white
    _rgb(185, 183, 180),  # 7  light gray
    _rgb(125, 123, 120),  # 8  mid gray
    _rgb( 70,  68,  66),  # 9  dark gray
    _rgb(125,  90,  55),  # 10 soil
    _rgb( 28,  28,  28),  # 11 near-black
], dtype=np.float32)
_reg(
    "sunflower_12",
    _sunflower_12, 6,
    "12 patches for sunflower photography: petals, disk, leaf, stem + neutral ramp (6x2)",
)



# ---------------------------------------------------------------------------
# Per-preset marker IDs (overrides the colorchecker_24 default of 0–3)
# ---------------------------------------------------------------------------

PRESET_MARKER_IDS: dict[str, dict[str, int]] = {
    "gray_50": _GRAY50_MARKER_IDS,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_preset(name: str) -> tuple[NDArray, int, str]:
    """Return (colors, cols, description) for a named preset."""
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Available: {available}")
    return PRESETS[name]


def list_presets() -> str:
    lines = [f"{'Name':<20}  {'Patches':>7}  {'Cols':>4}  Description"]
    lines.append("-" * 70)
    for name in sorted(PRESETS):
        colors, cols, desc = PRESETS[name]
        lines.append(f"{name:<20}  {len(colors):>7}  {cols:>4}  {desc}")
    return "\n".join(lines)
