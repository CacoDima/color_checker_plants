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
