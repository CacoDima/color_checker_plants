"""
Colour correction fitting.

Three methods, in increasing complexity:

1. 'matrix'  — 3×3 linear least-squares (fast, good for well-lit scenes).
2. 'poly'    — Cheung 2004 polynomial expansion: adds cross-channel and squared terms
               before the linear solve, handles moderate non-linearities.
3. 'root_poly' — Root-polynomial (Finlayson 2015): uses sqrt of products instead of
               products, better conditioned, less prone to overfitting.

The fitted model is a plain dict so it can be pickled / saved with np.save.

Reference:
  Cheung, V. et al. (2004). Successive rejection of defective colour-correction matrices.
  Finlayson, G. D., et al. (2015). Root-polynomial colour correction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_correction(
    measured: NDArray,
    reference: NDArray,
    method: str = "poly",
) -> dict:
    """
    Compute a colour correction model from measured patch colours to reference values.

    Parameters
    ----------
    measured  : (N, 3) float32 — colours sampled from the photo.
    reference : (N, 3) float32 — corresponding ground-truth reference colours.
    method    : 'matrix', 'poly', 'root_poly', or 'channel'.
                'channel' fits each RGB channel independently (R_out=f(R_in) only).
                Use this with achromatic-only training data (grayscale ramps) — the
                standard matrix/poly methods degenerate to greyscale when all reference
                patches have R==G==B.

    Returns
    -------
    model : dict with keys:
        'method'  : str
        'matrix'  : (K, 3) weight matrix  (None for 'channel')
        'expand'  : feature expansion fn  (None for 'channel')
        'scales'  : (3,) per-channel scale (only for 'channel')
        'biases'  : (3,) per-channel bias  (only for 'channel')
        'residuals': per-patch ΔE (approximate, in sRGB space)
    """
    measured  = np.asarray(measured,  dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if method == "channel":
        scales = np.zeros(3, dtype=np.float64)
        biases = np.zeros(3, dtype=np.float64)
        for c in range(3):
            X_c = np.column_stack([measured[:, c], np.ones(len(measured))])
            W_c, _, _, _ = np.linalg.lstsq(X_c, reference[:, c], rcond=None)
            scales[c], biases[c] = W_c
        predicted = measured * scales + biases
        residuals = np.sqrt(((predicted - reference) ** 2).sum(axis=1))
        return {
            "method":    "channel",
            "matrix":    None,
            "expand":    None,
            "scales":    scales.astype(np.float32),
            "biases":    biases.astype(np.float32),
            "residuals": residuals.astype(np.float32),
        }

    expand = _get_expand_fn(method)
    X = expand(measured)          # (N, K)
    # Solve X @ W = reference  →  W = pinv(X) @ reference
    W, _, _, _ = np.linalg.lstsq(X, reference, rcond=None)

    predicted  = X @ W
    residuals  = np.sqrt(((predicted - reference) ** 2).sum(axis=1))

    return {
        "method":    method,
        "matrix":    W.astype(np.float32),
        "expand":    expand,
        "scales":    None,
        "biases":    None,
        "residuals": residuals.astype(np.float32),
    }


def apply_correction_to_colors(colors: NDArray, model: dict) -> NDArray:
    """
    Apply a fitted correction model to an (N, 3) array of colours.

    Returns float32 values clipped to [0, 1].
    """
    if model["method"] == "channel":
        corrected = np.asarray(colors, dtype=np.float32) * model["scales"] + model["biases"]
        return np.clip(corrected, 0.0, 1.0).astype(np.float32)
    X = model["expand"](np.asarray(colors, dtype=np.float64))
    corrected = (X @ model["matrix"]).astype(np.float32)
    return np.clip(corrected, 0.0, 1.0)


def fit_summary(model: dict, patch_names: list[str] | None = None) -> str:
    """Human-readable per-patch residual report."""
    residuals = model["residuals"]
    lines = [
        f"Method : {model['method']}",
        f"Mean ΔE: {residuals.mean():.4f}",
        f"Max  ΔE: {residuals.max():.4f}",
        "",
        f"{'#':>3}  {'Patch':<20}  {'ΔE':>6}",
        "-" * 35,
    ]
    for i, r in enumerate(residuals):
        name = patch_names[i] if patch_names and i < len(patch_names) else f"patch_{i:02d}"
        lines.append(f"{i:>3}  {name:<20}  {r:>6.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feature expansion functions
# ---------------------------------------------------------------------------

def _get_expand_fn(method: str):
    if method == "matrix":
        return _expand_linear
    elif method == "poly":
        return _expand_cheung2004
    elif method == "root_poly":
        return _expand_root_poly
    else:
        raise ValueError(f"Unknown method {method!r}. Choose 'matrix', 'poly', or 'root_poly'.")


def _expand_linear(RGB: NDArray) -> NDArray:
    """[R, G, B, 1]  →  (N, 4)"""
    ones = np.ones((len(RGB), 1), dtype=RGB.dtype)
    return np.hstack([RGB, ones])


def _expand_cheung2004(RGB: NDArray) -> NDArray:
    """
    Cheung 2004 polynomial terms:
    [R, G, B, RG, RB, GB, R², G², B², RGB, 1]  →  (N, 11)
    """
    R, G, B = RGB[:, 0:1], RGB[:, 1:2], RGB[:, 2:3]
    ones = np.ones_like(R)
    return np.hstack([
        R, G, B,
        R * G, R * B, G * B,
        R ** 2, G ** 2, B ** 2,
        R * G * B,
        ones,
    ])


def _expand_root_poly(RGB: NDArray) -> NDArray:
    """
    Finlayson 2015 root-polynomial terms:
    [R, G, B, √(RG), √(RB), √(GB), √(R²G), √(R²B), √(G²R), √(G²B), √(B²R), √(B²G), 1]
    → (N, 13)
    """
    R, G, B = RGB[:, 0:1], RGB[:, 1:2], RGB[:, 2:3]
    eps = 1e-8
    ones = np.ones_like(R)
    return np.hstack([
        R, G, B,
        np.sqrt(np.abs(R * G) + eps),
        np.sqrt(np.abs(R * B) + eps),
        np.sqrt(np.abs(G * B) + eps),
        np.sqrt(np.abs(R ** 2 * G) + eps),
        np.sqrt(np.abs(R ** 2 * B) + eps),
        np.sqrt(np.abs(G ** 2 * R) + eps),
        np.sqrt(np.abs(G ** 2 * B) + eps),
        np.sqrt(np.abs(B ** 2 * R) + eps),
        np.sqrt(np.abs(B ** 2 * G) + eps),
        ones,
    ])
