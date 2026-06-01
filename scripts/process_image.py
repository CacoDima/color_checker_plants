"""
CLI: detect colour checker, fit correction, and apply it to one or more images.

Usage:
    # Detect, fit, correct a single image:
    python scripts/process_image.py checker_photo.jpg --output corrected.jpg

    # Apply a previously saved model to a batch of images:
    python scripts/process_image.py --model model.npz *.jpg

    # Save the fitted model for later reuse:
    python scripts/process_image.py checker_photo.jpg --save-model model.npz

    # Choose fitting method:
    python scripts/process_image.py checker_photo.jpg --method poly

    # Show debug view (marker detection overlay):
    python scripts/process_image.py checker_photo.jpg --debug
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from color_checker_plants import (
    detect_color_checker,
    fit_correction,
    apply_correction,
    fit_summary,
)
from color_checker_plants.templates import PATCH_NAMES
from color_checker_plants.utils import load_image, save_image


def process_single(args):
    print(f"Loading {args.input} …")
    image = load_image(args.input)

    print("Detecting colour checker …")
    result = detect_color_checker(
        image,
        extraction_method=args.extraction,
        sigma=args.sigma,
    )

    if result is None:
        print("ERROR: Could not find all 4 ArUco corner markers in the image.")
        print("Make sure IDs 0, 1, 2, 3 are all visible and not occluded.")
        sys.exit(1)

    print(f"  Detected {len(result.measured_colors)} patches.")

    if args.debug:
        debug_path = Path(args.input).stem + "_debug.jpg"
        cv2.imwrite(debug_path, result.debug_image)
        warped_path = Path(args.input).stem + "_warped.jpg"
        save_image(warped_path, result.warped_image)
        print(f"  Saved debug overlay → {debug_path}")
        print(f"  Saved warped grid   → {warped_path}")

    print(f"Fitting colour correction (method={args.method}) …")
    model = fit_correction(result.measured_colors, result.reference_colors, method=args.method)

    print(fit_summary(model, PATCH_NAMES))

    if args.save_model:
        _save_model(model, args.save_model)
        print(f"  Model saved → {args.save_model}")

    print(f"Applying correction to {args.input} …")
    corrected = apply_correction(image, model)
    out_path = args.output or Path(args.input).stem + "_corrected.jpg"
    save_image(out_path, corrected)
    print(f"  Corrected image → {out_path}")

    return model


def apply_saved_model(args):
    model = _load_model(args.model)
    print(f"Loaded model from {args.model}  (method={model['method']})")
    for src in args.inputs:
        image = load_image(src)
        corrected = apply_correction(image, model)
        out_path = Path(src).stem + "_corrected" + Path(src).suffix
        save_image(out_path, corrected)
        print(f"  {src} → {out_path}")


def _save_model(model: dict, path: str):
    np.savez(
        path,
        method=np.array([model["method"]]),
        matrix=model["matrix"],
        residuals=model["residuals"],
    )


def _load_model(path: str) -> dict:
    from color_checker_plants.fitting import _get_expand_fn
    data = np.load(path, allow_pickle=False)
    method = str(data["method"][0])
    return {
        "method":    method,
        "matrix":    data["matrix"],
        "expand":    _get_expand_fn(method),
        "residuals": data["residuals"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect colour checker, fit, and apply colour correction."
    )
    sub = parser.add_subparsers(dest="cmd")

    # --- detect + fit + correct ---
    p_detect = sub.add_parser("detect", help="Detect checker in one image and fit a model.")
    p_detect.add_argument("input",  help="Photo containing the colour checker.")
    p_detect.add_argument("--output",      help="Output corrected image path.")
    p_detect.add_argument("--method",      default="poly",
                          choices=["matrix", "poly", "root_poly"],
                          help="Fitting method (default: poly).")
    p_detect.add_argument("--extraction",  default="sigma_clip",
                          choices=["sigma_clip", "median", "trimmed_mean"])
    p_detect.add_argument("--sigma",       type=float, default=2.5)
    p_detect.add_argument("--save-model",  metavar="FILE",
                          help="Save fitted model to .npz for batch use.")
    p_detect.add_argument("--debug",       action="store_true",
                          help="Save marker overlay and warped grid images.")

    # --- apply saved model ---
    p_apply = sub.add_parser("apply", help="Apply a saved model to a batch of images.")
    p_apply.add_argument("--model",  required=True, help="Path to .npz model file.")
    p_apply.add_argument("inputs",  nargs="+", help="Images to correct.")

    # Allow calling without subcommand for convenience: process_image.py photo.jpg
    parser.add_argument("input",   nargs="?")
    parser.add_argument("--output")
    parser.add_argument("--method", default="poly", choices=["matrix", "poly", "root_poly"])
    parser.add_argument("--extraction", default="sigma_clip",
                        choices=["sigma_clip", "median", "trimmed_mean"])
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--save-model", metavar="FILE")
    parser.add_argument("--model")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("inputs", nargs="*")

    args = parser.parse_args()

    # Subcommand routing
    if args.cmd == "detect":
        process_single(args)
    elif args.cmd == "apply":
        apply_saved_model(args)
    elif args.input:
        # No subcommand but positional arg given → detect mode
        process_single(args)
    elif args.model and args.inputs:
        apply_saved_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()