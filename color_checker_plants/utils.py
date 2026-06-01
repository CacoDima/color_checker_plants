"""Miscellaneous utilities: image I/O, colour metrics, profile save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

_RAW_EXTENSIONS = {'.cr2', '.cr3', '.nef', '.arw', '.orf', '.rw2', '.dng', '.raf', '.raw', '.pef', '.srw'}


def load_image(path: str) -> NDArray:
    """
    Load image as float32 RGB [0, 1].

    Supported formats
    -----------------
    - JPEG, PNG                  — 8-bit, via OpenCV
    - TIFF                       — 8-bit and 16-bit, via OpenCV
    - RAW (CR2, CR3, NEF, ARW,
           ORF, RW2, DNG, RAF …) — via rawpy (``pip install rawpy``)
                                   Camera white balance is applied automatically.
    """
    import cv2
    p = Path(path)

    if p.suffix.lower() in _RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError:
            raise ImportError(
                "rawpy is required for RAW files.  Install with:  pip install rawpy"
            )
        with rawpy.imread(str(p)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=16,
                no_auto_bright=True,
            )
        return rgb.astype(np.float32) / 65535.0

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def save_image(path: str, image: NDArray, bit_depth: int = 8) -> None:
    """
    Save float32 RGB [0, 1] image to disk.

    Parameters
    ----------
    path      : Output path. Format is inferred from extension (.jpg, .png, .tif, .tiff …)
    image     : (H, W, 3) float32 RGB array, values in [0, 1].
    bit_depth : 8 (default) or 16. 16-bit is only meaningful for TIFF output.
    """
    import cv2
    img = np.clip(np.asarray(image), 0.0, 1.0)
    if img.ndim == 3 and img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = img

    if bit_depth == 16:
        cv2.imwrite(str(path), (bgr * 65535).astype(np.uint16))
    else:
        cv2.imwrite(str(path), (bgr * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Colour metrics
# ---------------------------------------------------------------------------

def delta_e_simple(a: NDArray, b: NDArray) -> NDArray:
    """Euclidean distance in sRGB space (not perceptually uniform). Fast."""
    return np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum(axis=-1))


def srgb_to_linear(srgb: NDArray) -> NDArray:
    """sRGB inverse EOTF: non-linear sRGB [0,1] → linear light [0,1]."""
    srgb = np.asarray(srgb, dtype=np.float32)
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb(linear: NDArray) -> NDArray:
    """sRGB EOTF: linear light [0,1] → non-linear sRGB [0,1]."""
    linear = np.asarray(linear, dtype=np.float32)
    return np.clip(
        np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * linear ** (1.0 / 2.4) - 0.055,
        ),
        0.0, 1.0,
    ).astype(np.float32)


def _srgb_to_lab(srgb: NDArray) -> NDArray:
    """sRGB [0,1] → CIE L*a*b* (D65 illuminant). Internal helper."""
    lin = np.where(
        srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4
    )
    # sRGB (D65) → XYZ
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float64)
    xyz = lin @ M.T
    # Normalise to D65 white point
    xyz = xyz / np.array([0.95047, 1.00000, 1.08883])
    # XYZ → f(t)
    eps, kappa = 0.008856, 903.3
    f = np.where(xyz > eps, xyz ** (1.0 / 3.0), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e_2000(a: NDArray, b: NDArray) -> NDArray:
    """
    CIE ΔE2000 between two sRGB colour arrays.

    Parameters
    ----------
    a, b : (..., 3) float sRGB [0, 1]

    Returns
    -------
    (...,) float32 ΔE2000 values.
    Perceptible difference threshold ≈ 1.0; excellent correction < 2.0.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lab1 = _srgb_to_lab(a)
    lab2 = _srgb_to_lab(b)

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_avg7 = ((C1 + C2) / 2) ** 7
    G  = 0.5 * (1.0 - np.sqrt(C_avg7 / (C_avg7 + 25.0 ** 7)))
    a1p, a2p = a1 * (1.0 + G), a2 * (1.0 + G)
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = np.where(
        np.abs(h2p - h1p) <= 180.0,
        h2p - h1p,
        np.where(h2p <= h1p, h2p - h1p + 360.0, h2p - h1p - 360.0),
    )
    dhp  = np.where(C1p * C2p == 0.0, 0.0, dhp)
    dHp  = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0
    hp_avg = np.where(
        np.abs(h1p - h2p) > 180.0,
        np.where(h1p + h2p < 360.0, (h1p + h2p + 360.0) / 2.0, (h1p + h2p - 360.0) / 2.0),
        (h1p + h2p) / 2.0,
    )
    hp_avg = np.where(C1p * C2p == 0.0, h1p + h2p, hp_avg)

    T = (1.0
         - 0.17 * np.cos(np.radians(hp_avg - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hp_avg))
         + 0.32 * np.cos(np.radians(3.0 * hp_avg + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hp_avg - 63.0)))

    SL = 1.0 + 0.015 * (Lp_avg - 50.0) ** 2 / np.sqrt(20.0 + (Lp_avg - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T

    Cp7 = Cp_avg ** 7
    RC  = 2.0 * np.sqrt(Cp7 / (Cp7 + 25.0 ** 7))
    RT  = -np.sin(np.radians(60.0 * np.exp(-((hp_avg - 275.0) / 25.0) ** 2))) * RC

    de = np.sqrt(
        (dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )
    return de.astype(np.float32)


# ---------------------------------------------------------------------------
# Photo profiles (save / load fitted correction model)
# ---------------------------------------------------------------------------

def save_profile(model: dict, path: str) -> None:
    """
    Save a fitted colour correction model to a .npz file.

    The profile stores the method name and all numeric weights so it can
    be loaded later and applied to any image taken under the same lighting.

    Example
    -------
        model = fit_correction(result.measured_colors, result.reference_colors)
        save_profile(model, "studio_light.npz")
    """
    data: dict[str, np.ndarray] = {"method": np.array(model["method"])}
    if model.get("residuals") is not None:
        data["residuals"] = model["residuals"]
    if model.get("matrix") is not None:
        data["matrix"] = model["matrix"]
    if model.get("scales") is not None:
        data["scales"] = model["scales"]
        data["biases"] = model["biases"]
    np.savez(str(path), **data)


def load_profile(path: str) -> dict:
    """
    Load a colour correction profile saved with ``save_profile()``.

    Returns a model dict accepted by ``apply_correction()`` and
    ``apply_correction_to_colors()``.

    Example
    -------
        model = load_profile("studio_light.npz")
        corrected = apply_correction(image, model)
    """
    from .fitting import _get_expand_fn
    npz   = np.load(str(path))
    method = str(npz["method"])
    model: dict = {
        "method":    method,
        "residuals": npz["residuals"] if "residuals" in npz else None,
        "matrix":    npz["matrix"]    if "matrix"    in npz else None,
        "expand":    _get_expand_fn(method) if method != "channel" else None,
        "scales":    npz["scales"]    if "scales"    in npz else None,
        "biases":    npz["biases"]    if "biases"    in npz else None,
    }
    return model
