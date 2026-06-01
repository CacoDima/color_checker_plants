"""
ArUco marker detection and corner extraction.

Physical card layout:
  ┌─[M0]──────────────[M1]─┐
  │                        │
  │   colour grid (6 × 4)  │
  │                        │
  └─[M3]──────────────[M2]─┘

M0=top-left, M1=top-right, M2=bottom-right, M3=bottom-left.
The inner corner of each marker is the reference point for perspective correction.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np
from numpy.typing import NDArray

from .templates.colorchecker_24 import MARKER_IDS, MARKER_INNER_CORNER_INDEX

# DICT_5X5_50 gives a Hamming distance of 8 between any two codes,
# vs. 6 for DICT_4X4_50 — meaningfully more robust to print artefacts and blur.
# 50 IDs is more than enough (we only use 4).
_DICT_ID = cv2.aruco.DICT_5X5_50

try:
    # OpenCV >= 4.7 moved aruco to cv2.aruco directly
    _ARUCO_DICT = cv2.aruco.getPredefinedDictionary(_DICT_ID)
    _ARUCO_PARAMS = cv2.aruco.DetectorParameters()
    _USE_NEW_API = True
except AttributeError:
    _ARUCO_DICT = cv2.aruco.Dictionary_get(_DICT_ID)
    _ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
    _USE_NEW_API = False


def generate_marker_image(marker_id: int, size_px: int = 200) -> NDArray:
    """Return a grayscale image of an ArUco marker."""
    img = np.zeros((size_px, size_px), dtype=np.uint8)
    if _USE_NEW_API:
        img = cv2.aruco.generateImageMarker(_ARUCO_DICT, marker_id, size_px, img, 1)
    else:
        cv2.aruco.drawMarker(_ARUCO_DICT, marker_id, size_px, img, 1)
    return img


def detect_markers(image: NDArray) -> tuple[list[NDArray], NDArray]:
    """
    Detect all ArUco markers in the image.

    Returns
    -------
    corners : list of (1, 4, 2) float32 arrays, one per detected marker.
              Each inner array holds [TL, TR, BR, BL] pixel coords.
    ids     : (N, 1) int array of detected marker IDs, or None if none found.
    """
    gray = _to_gray(image)

    if _USE_NEW_API:
        detector = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, _ARUCO_DICT, parameters=_ARUCO_PARAMS)

    return corners, ids


def extract_grid_corners(
    corners: list[NDArray],
    ids: NDArray,
    marker_ids: dict[str, int] | None = None,
) -> NDArray | None:
    """
    From detected markers, extract the 4 inner corners that define the colour grid.

    Supports 3-marker fallback: if exactly one marker is missing the fourth
    corner is extrapolated assuming the card is approximately a parallelogram
    (A + C = B + D).  A UserWarning is emitted in that case.

    Parameters
    ----------
    corners    : Output from detect_markers().
    ids        : Output from detect_markers().
    marker_ids : Mapping from position name to marker ID.
                 Defaults to MARKER_IDS from the 24-patch template.

    Returns
    -------
    grid_corners : (4, 2) float32 array ordered [TL, TR, BR, BL],
                   or None if fewer than 3 markers were found.
    """
    if ids is None:
        return None

    if marker_ids is None:
        marker_ids = MARKER_IDS

    _POSITIONS = ("top_left", "top_right", "bottom_right", "bottom_left")

    ids_flat  = ids.flatten()
    corner_map: dict[str, NDArray] = {}
    missing: list[str] = []

    for position in _POSITIONS:
        mid     = marker_ids[position]
        matches = np.where(ids_flat == mid)[0]
        if len(matches) == 0:
            missing.append(position)
            continue
        marker_corners = corners[matches[0]][0]  # (4, 2): TL, TR, BR, BL
        corner_map[position] = marker_corners[MARKER_INNER_CORNER_INDEX[position]]

    if len(missing) == 0:
        pass  # all 4 found — nothing to do

    elif len(missing) == 1:
        # Parallelogram rule: TL + BR = TR + BL  →  missing = sum of opposite - adjacent
        pos = missing[0]
        TL = corner_map.get("top_left")
        TR = corner_map.get("top_right")
        BR = corner_map.get("bottom_right")
        BL = corner_map.get("bottom_left")
        if   pos == "top_left":     corner_map["top_left"]     = TR + BL - BR
        elif pos == "top_right":    corner_map["top_right"]    = TL + BR - BL
        elif pos == "bottom_right": corner_map["bottom_right"] = TR + BL - TL
        elif pos == "bottom_left":  corner_map["bottom_left"]  = TL + BR - TR
        warnings.warn(
            f"Marker '{pos}' not found — extrapolating 4th corner from 3 markers. "
            "Accuracy may be reduced near that corner.",
            UserWarning, stacklevel=3,
        )

    else:
        return None  # 2+ markers missing — cannot recover

    grid_corners = np.array([
        corner_map["top_left"],
        corner_map["top_right"],
        corner_map["bottom_right"],
        corner_map["bottom_left"],
    ], dtype=np.float32)

    return grid_corners


def draw_markers(image: NDArray, corners: list[NDArray], ids: NDArray) -> NDArray:
    """Return a copy of the image with detected markers outlined."""
    vis = image.copy()
    if ids is not None:
        if _USE_NEW_API:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        else:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    return vis


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_gray(image: NDArray) -> NDArray:
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return gray
