"""
color-checker-plants
====================

Robust colour-checker detection using ArUco corner markers, perspective
correction, sigma-clipped colour extraction, and polynomial colour fitting.

Quick start
-----------
    from color_checker_plants import detect_color_checker, fit_correction, apply_correction
    from color_checker_plants.utils import load_image, save_image

    image = load_image("photo.jpg")

    result = detect_color_checker(image)
    if result is None:
        raise RuntimeError("Colour checker not found — are all 4 ArUco markers visible?")

    model = fit_correction(result.measured_colors, result.reference_colors, method="poly")
    corrected = apply_correction(image, model)
    save_image("corrected.jpg", corrected)
"""

from .detection  import detect_color_checker, DetectionResult
from .fitting    import fit_correction, apply_correction_to_colors, fit_summary
from .correction import apply_correction, correction_gain_map
from .extraction import extract_grid_colors
from .markers    import generate_marker_image, detect_markers, extract_grid_corners

__version__ = "0.1.0"

__all__ = [
    "detect_color_checker",
    "DetectionResult",
    "fit_correction",
    "apply_correction_to_colors",
    "fit_summary",
    "apply_correction",
    "correction_gain_map",
    "extract_grid_colors",
    "generate_marker_image",
    "detect_markers",
    "extract_grid_corners",
]
