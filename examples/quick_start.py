"""
Quick-start example: full pipeline from photo to corrected image.

Run from the project root:
    python examples/quick_start.py path/to/photo_with_checker.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from color_checker_plants import (
    detect_color_checker,
    fit_correction,
    apply_correction,
    fit_summary,
)
from color_checker_plants.utils import load_image, save_image, delta_e_simple
from color_checker_plants.templates import PATCH_NAMES


def main(photo_path: str):
    # ── 1. Load image ────────────────────────────────────────────────
    print(f"Loading {photo_path}")
    image = load_image(photo_path)

    # ── 2. Detect colour checker ─────────────────────────────────────
    print("Detecting colour checker …")
    result = detect_color_checker(
        image,
        extraction_method="sigma_clip",   # robust to dust/highlights
        sigma=2.5,
    )

    if result is None:
        print(
            "Could not find all 4 ArUco markers.\n"
            "Tips:\n"
            "  • Make sure all four markers (IDs 0–3) are fully in frame.\n"
            "  • Avoid heavy motion blur or extreme under-exposure.\n"
            "  • Try shooting the checker on a contrasting background."
        )
        sys.exit(1)

    print(f"Found {len(result.measured_colors)} patches.")

    # Save debug view
    stem = Path(photo_path).stem
    import cv2
    cv2.imwrite(f"{stem}_debug.jpg", result.debug_image)
    save_image(f"{stem}_warped.jpg", result.warped_image)
    print(f"  Debug overlay → {stem}_debug.jpg")
    print(f"  Warped grid   → {stem}_warped.jpg")

    # ── 3. Fit colour correction ──────────────────────────────────────
    print("\nFitting polynomial correction …")
    model = fit_correction(
        result.measured_colors,
        result.reference_colors,
        method="poly",       # 'matrix' | 'poly' | 'root_poly'
    )

    print(fit_summary(model, PATCH_NAMES))

    # Raw ΔE before correction
    de_before = delta_e_simple(result.measured_colors, result.reference_colors)
    print(f"\nΔE before correction: mean={de_before.mean():.4f}  max={de_before.max():.4f}")

    # ── 4. Apply correction to the original image ─────────────────────
    corrected = apply_correction(image, model)

    # ΔE after correction (re-extract patches from corrected image to verify)
    from color_checker_plants import detect_color_checker as _detect
    result2 = _detect(corrected)
    if result2 is not None:
        de_after = delta_e_simple(result2.measured_colors, result2.reference_colors)
        print(f"ΔE after  correction: mean={de_after.mean():.4f}  max={de_after.max():.4f}")

    out_path = f"{stem}_corrected.jpg"
    save_image(out_path, corrected)
    print(f"\nCorrected image → {out_path}")

    # ── 5. Optionally save model for batch processing ─────────────────
    model_path = f"{stem}_model.npz"
    np.savez(
        model_path,
        method=np.array([model["method"]]),
        matrix=model["matrix"],
        residuals=model["residuals"],
    )
    print(f"Model saved → {model_path}")
    print(
        "\nTo apply this model to other images from the same shoot:\n"
        f"    python scripts/process_image.py apply --model {model_path} *.jpg"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/quick_start.py <photo.jpg>")
        sys.exit(1)
    main(sys.argv[1])
