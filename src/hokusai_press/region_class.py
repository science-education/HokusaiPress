"""Pure image-patch classification for rich region handling.

This module is intentionally self-contained: it does not know about
RegionKind, OCR, layout providers, or pipeline state. It classifies only the
pixels of a cropped region image.
"""

from __future__ import annotations

import cv2
import numpy as np

# Patches smaller than this are too unstable for threshold/connected-component
# statistics. Treat them as unknown rather than overfitting noise.
MIN_PATCH_PIXELS = 64

# Almost-white, near-uniform crops carry no useful region class.
BLANK_MEAN_MIN = 245.0
BLANK_STD_MAX = 3.0

# Canny thresholds for the edge-density feature. These are deliberately broad:
# classification relies on relative density, not precise edge localization.
CANNY_LOW = 50
CANNY_HIGH = 150

# Solid fills are dense ink/tint panels with little internal edge structure and
# near-uniform tone. The fill threshold is low enough to catch mid-gray panels
# after Otsu marks the darker side of the tone as ink.
SOLID_FILL_MIN = 0.50
SOLID_EDGE_MAX = 0.08
SOLID_STD_MAX = 10.0
SOLID_OCCUPIED_BINS_MAX = 3

# Photos/halftones have a broad tone distribution and are not almost pure
# black/white. A patch may satisfy either gray-level diversity or std spread.
PHOTO_OCCUPIED_BINS_MIN = 8
PHOTO_STD_MIN = 22.0
PHOTO_EXTREME_MAX = 0.92

# Line art has sparse/moderate ink but a noticeable edge footprint from strokes,
# rules, or outlines.
LINE_FILL_MAX = 0.45
LINE_EDGE_MIN = 0.015

# Text is a secondary heuristic: many similarly sized connected components,
# usually arranged on at least one horizontal text line.
TEXT_MIN_COMPONENTS = 8
TEXT_HEIGHT_CV_MAX = 0.60
TEXT_LINE_Y_TOLERANCE = 0.65


def classify_patch(patch) -> str:
    """Classify a cropped region image.

    Args:
        patch: An HxW or HxWx3 uint8 numpy array.

    Returns:
        One of: "photo", "solid_fill", "line_art", "text", "other".
    """
    gray = _gray_u8(patch)
    if gray is None or gray.size < MIN_PATCH_PIXELS:
        return "other"

    h, w = gray.shape
    total = h * w
    std = float(gray.std())
    occupied_bins = _occupied_bins(gray)
    extreme_ratio = float(((gray <= 8) | (gray >= 247)).sum()) / total
    if float(gray.mean()) >= BLANK_MEAN_MIN and std <= BLANK_STD_MAX:
        return "other"

    threshold = _otsu_threshold(gray)
    ink = (gray <= threshold).astype(np.uint8)
    fill_ratio = float(ink.mean())

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    edge_density = float((edges > 0).mean())

    if (
        (occupied_bins >= PHOTO_OCCUPIED_BINS_MIN or std >= PHOTO_STD_MIN)
        and extreme_ratio <= PHOTO_EXTREME_MAX
        and not _near_uniform(gray, std, occupied_bins)
    ):
        return "photo"

    if (
        fill_ratio >= SOLID_FILL_MIN
        and edge_density < SOLID_EDGE_MAX
        and _near_uniform(gray, std, occupied_bins)
    ):
        return "solid_fill"

    if fill_ratio <= LINE_FILL_MAX and edge_density >= LINE_EDGE_MIN:
        return "line_art"

    if _looks_like_text(ink):
        return "text"

    return "other"


def _gray_u8(patch) -> np.ndarray | None:
    if not isinstance(patch, np.ndarray) or patch.size == 0:
        return None
    if patch.dtype != np.uint8:
        return None
    if patch.ndim == 2:
        return patch
    if patch.ndim == 3 and patch.shape[2] == 3:
        return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return None


def _otsu_threshold(gray: np.ndarray) -> int:
    if int(gray.min()) == int(gray.max()):
        return int(gray.max())
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(threshold)


def _occupied_bins(gray: np.ndarray) -> int:
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    return int((hist > gray.size * 0.005).sum())


def _near_uniform(gray: np.ndarray, std: float, occupied_bins: int) -> bool:
    return std <= SOLID_STD_MAX or occupied_bins <= SOLID_OCCUPIED_BINS_MAX


def _looks_like_text(ink: np.ndarray) -> bool:
    n, _, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= TEXT_MIN_COMPONENTS:
        return False

    heights: list[int] = []
    centers_y: list[float] = []
    page_area = ink.shape[0] * ink.shape[1]
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 2 or area > page_area * 0.03:
            continue
        if h < 3 or w < 1:
            continue
        heights.append(int(h))
        centers_y.append(float(centroids[i][1]))

    if len(heights) < TEXT_MIN_COMPONENTS:
        return False

    hs = np.asarray(heights, dtype=np.float32)
    mean_h = float(hs.mean())
    if mean_h <= 0:
        return False
    if float(hs.std()) / mean_h > TEXT_HEIGHT_CV_MAX:
        return False

    ys = np.asarray(centers_y, dtype=np.float32)
    line_tol = max(2.0, mean_h * TEXT_LINE_Y_TOLERANCE)
    for y in ys:
        if int((np.abs(ys - y) <= line_tol).sum()) >= TEXT_MIN_COMPONENTS:
            return True
    return False
