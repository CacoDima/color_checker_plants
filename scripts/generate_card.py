"""
Generate a printable colour checker card as a PNG.

The card has:
  - 4 ArUco markers (DICT_5X5_50, IDs 0–3) at the corners, one per corner.
  - A customisable grid of colour patches in the centre.
  - A white border around everything.

Usage examples:
    # Default X-Rite 24-patch Classic
    python scripts/generate_card.py --output card.png --dpi 300

    # Named preset
    python scripts/generate_card.py --preset white_grad --output white.png

    # Custom colours from a JSON file  (values 0-255 or 0-1, auto-detected)
    # File format: [[R,G,B], [R,G,B], ...]
    python scripts/generate_card.py --colors-file my_colors.json --cols 6

    # List all built-in presets
    python scripts/generate_card.py --list-presets

Physical recommendation:
  - Print on matte photo paper (glossy can cause reflections in photos).
  - Laminate if using outdoors.
  - A4 (210 mm wide) at 300 DPI gives correct physical dimensions.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from color_checker_plants.markers import generate_marker_image
from color_checker_plants.templates.colorchecker_24 import MARKER_IDS as _DEFAULT_MARKER_IDS
from color_checker_plants.templates.presets import get_preset, list_presets, PRESETS, PRESET_MARKER_IDS


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_card(
    colors: np.ndarray,
    cols: int,
    dpi: int = 300,
    width_mm: float = 210.0,
    border_mm: float = 10.0,
    marker_size_mm: float = 20.0,
    gap_mm: float = 3.0,
    show_labels: bool = False,
    label_names: list[str] | None = None,
    marker_ids: dict[str, int] | None = None,
) -> np.ndarray:
    """
    Build and return the card as a uint8 RGB image.

    Parameters
    ----------
    colors       : (N, 3) float32 sRGB array, values in [0, 1].
    cols         : Number of columns in the colour grid.
                   Rows are computed as ceil(N / cols).
    dpi          : Output resolution. Print at exactly this DPI.
    width_mm     : Card width in millimetres.
    border_mm    : White margin on every edge (mm).
    marker_size_mm: Side length of each ArUco marker (mm).
    gap_mm       : Gap between marker and colour grid (mm).
    show_labels  : Overlay patch index or name on each cell.
    label_names  : Optional list of strings, one per patch. If None and
                   show_labels is True, zero-based indices are used.

    Layout
    ------
      ┌── border ────────────────────────────────────────────────────┐
      │  [M0 ID=0]  gap  ──────── colour grid ────────  gap  [M1 ID=1] │
      │                                                               │
      │  [M3 ID=3]  gap  ──────────────────────────  gap  [M2 ID=2] │
      └───────────────────────────────────────────────────────────────┘
    """
    colors = np.asarray(colors, dtype=np.float32)
    n_patches = len(colors)
    rows = math.ceil(n_patches / cols)

    def mm2px(mm: float) -> int:
        return int(round(mm / 25.4 * dpi))

    border = mm2px(border_mm)
    msize  = mm2px(marker_size_mm)
    gap    = mm2px(gap_mm)
    card_w = mm2px(width_mm)

    grid_w = card_w - 2 * border - 2 * msize - 2 * gap
    cell_w = grid_w // cols
    cell_h = cell_w  # square cells
    grid_h = cell_h * rows

    card_h = 2 * border + msize + gap + grid_h + gap + msize
    card_h = max(card_h, mm2px(width_mm / 1.41))

    canvas = np.full((card_h, card_w, 3), 255, dtype=np.uint8)

    grid_x0 = border + msize + gap
    grid_y0 = border + msize + gap

    # ── Colour patches ────────────────────────────────────────────────
    for idx in range(n_patches):
        row = idx // cols
        col = idx % cols
        c = colors[idx]
        bgr = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
        x0 = grid_x0 + col * cell_w
        y0 = grid_y0 + row * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), bgr, -1)

        if show_labels:
            text = label_names[idx] if label_names and idx < len(label_names) else str(idx)
            _draw_label(canvas, text, x0, y0, cell_w, cell_h, c)

    # Fill any trailing empty cells with light grey so the grid outline is complete
    for idx in range(n_patches, rows * cols):
        row = idx // cols
        col = idx % cols
        x0 = grid_x0 + col * cell_w
        y0 = grid_y0 + row * cell_h
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (230, 230, 230), -1)

    # ── Grid border ───────────────────────────────────────────────────
    cv2.rectangle(
        canvas,
        (grid_x0, grid_y0),
        (grid_x0 + grid_w - 1, grid_y0 + grid_h - 1),
        (160, 160, 160),
        1,
    )

    # ── ArUco markers ─────────────────────────────────────────────────
    if marker_ids is None:
        marker_ids = _DEFAULT_MARKER_IDS
    corners_px = {
        "top_left":     (border,                   border),
        "top_right":    (card_w - border - msize,  border),
        "bottom_right": (card_w - border - msize,  card_h - border - msize),
        "bottom_left":  (border,                   card_h - border - msize),
    }
    for position, mid in marker_ids.items():
        marker_img = generate_marker_image(mid, msize)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        mx, my = corners_px[position]
        canvas[my : my + msize, mx : mx + msize] = marker_bgr

    return canvas


# ---------------------------------------------------------------------------
# Label rendering
# ---------------------------------------------------------------------------

def _draw_label(
    canvas: np.ndarray,
    text: str,
    x0: int,
    y0: int,
    cell_w: int,
    cell_h: int,
    patch_color: np.ndarray,
) -> None:
    """Draw a small index/name label on a patch, choosing black or white ink."""
    luminance = 0.299 * patch_color[0] + 0.587 * patch_color[1] + 0.114 * patch_color[2]
    ink = (0, 0, 0) if luminance > 0.45 else (255, 255, 255)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.25, cell_w / 300.0)
    thickness  = 1

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = x0 + (cell_w - tw) // 2
    ty = y0 + (cell_h + th) // 2
    cv2.putText(canvas, text, (tx, ty), font, font_scale, ink, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Colour loading
# ---------------------------------------------------------------------------

def load_colors_from_file(path: str) -> np.ndarray:
    """
    Load custom colours from a JSON or CSV file.

    JSON format: [[R, G, B], [R, G, B], ...]
      Values may be integers 0–255 or floats 0–1 (auto-detected by magnitude).

    CSV format: one patch per line, three comma-separated values.
      Same auto-detection for 0–255 vs 0–1.
    """
    p = Path(path)
    if p.suffix.lower() in (".json",):
        raw = json.loads(p.read_text())
        colors = np.array(raw, dtype=np.float32)
    elif p.suffix.lower() in (".csv", ".txt"):
        colors = np.loadtxt(path, delimiter=",", dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file extension: {p.suffix}. Use .json or .csv")

    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {colors.shape}")

    # Auto-detect 0-255 encoding
    if colors.max() > 1.5:
        colors = colors / 255.0

    return np.clip(colors, 0.0, 1.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a printable ArUco-framed colour checker card.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_card.py --output card.png
  python scripts/generate_card.py --preset white_grad --output white.png
  python scripts/generate_card.py --preset gray_24 --labels --output gray.png
  python scripts/generate_card.py --colors-file my_colors.json --cols 4 --output custom.png
  python scripts/generate_card.py --list-presets
""",
    )

    # Colour source (mutually exclusive)
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument(
        "--preset", default="classic",
        metavar="NAME",
        help="Named colour preset (default: classic). See --list-presets.",
    )
    color_group.add_argument(
        "--colors-file",
        metavar="FILE",
        help="JSON or CSV file with custom patch colours [[R,G,B], ...] (0-255 or 0-1).",
    )
    color_group.add_argument(
        "--list-presets", action="store_true",
        help="Print all available presets and exit.",
    )

    parser.add_argument("--cols",       type=int,   default=None,
                        help="Number of columns in the grid (auto from preset if omitted).")
    parser.add_argument("--output",     default="color_checker_card.png")
    parser.add_argument("--dpi",        type=int,   default=300)
    parser.add_argument("--width_mm",   type=float, default=210.0,
                        help="Card width in mm (default: A4 width = 210 mm)")
    parser.add_argument("--border_mm",  type=float, default=10.0)
    parser.add_argument("--marker_mm",  type=float, default=20.0,
                        help="Size of each ArUco marker in mm")
    parser.add_argument("--labels",     action="store_true",
                        help="Overlay patch index on each cell.")

    args = parser.parse_args()

    if args.list_presets:
        print(list_presets())
        return

    # ── Resolve colour source ─────────────────────────────────────────
    label_names: list[str] | None = None
    card_marker_ids: dict[str, int] | None = None

    if args.colors_file:
        colors = load_colors_from_file(args.colors_file)
        cols   = args.cols or _auto_cols(len(colors))
        print(f"Loaded {len(colors)} patches from {args.colors_file}")
    else:
        name = args.preset
        colors, preset_cols, desc = get_preset(name)
        cols = args.cols or preset_cols
        card_marker_ids = PRESET_MARKER_IDS.get(name)
        print(f"Preset: {name} — {desc}")
        print(f"Patches: {len(colors)}  Columns: {cols}")
        if card_marker_ids:
            print(f"Marker IDs: {list(card_marker_ids.values())}")

    # ── Build and save ────────────────────────────────────────────────
    card = build_card(
        colors=colors,
        cols=cols,
        dpi=args.dpi,
        width_mm=args.width_mm,
        border_mm=args.border_mm,
        marker_size_mm=args.marker_mm,
        show_labels=args.labels,
        label_names=label_names,
        marker_ids=card_marker_ids,
    )

    out_bgr = cv2.cvtColor(card, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, out_bgr)
    print(f"Saved {args.output}  ({card.shape[1]}x{card.shape[0]} px, {args.dpi} DPI)")
    print(f"Print at {args.dpi} DPI for correct physical dimensions.")


def _auto_cols(n: int) -> int:
    """Pick the most square-ish number of columns for n patches."""
    best = 1
    best_ratio = float("inf")
    for c in range(1, n + 1):
        r = math.ceil(n / c)
        ratio = max(c, r) / min(c, r)
        if ratio < best_ratio:
            best_ratio = ratio
            best = c
    # Prefer landscape (cols >= rows)
    if best < math.ceil(n / best):
        best = math.ceil(n / best)
    return best


if __name__ == "__main__":
    main()