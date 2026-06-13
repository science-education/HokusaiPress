"""Automatic deskew with a confidence score.

Algorithm (in the spirit of ScanTailor's imageproc/SkewFinder, GPLv3): for a
range of candidate angles, shear-rotate the binarized page and measure how
"peaky" its projection profile is — a correctly deskewed page of text has
sharp, well-separated text-line ridges, so the sum of squared first
differences of the projection is maximized at the true skew angle.

Writing-direction agnostic: the score is the better of the horizontal
(yokogaki rows) and vertical (tategaki columns) projection sharpness, so the
same routine deskews both. Coarse-to-fine search keeps it fast.

Confidence is how far the best angle's sharpness stands above the spread of
scores: (best - median) / (median - min). This is invariant to an additive
baseline, so a large illustration/halftone (which adds a roughly
angle-independent offset to every projection) no longer deflates the score of a
page whose text lines still peak sharply — fixing false "low confidence" flags
on figure+text pages. Below GOOD_CONFIDENCE the page is flagged for review.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..model import Deskew

MAX_ANGLE = 7.0          # degrees; OCR tolerates only a few degrees of skew
COARSE_STEP = 0.5
FINE_STEP = 0.05
GOOD_CONFIDENCE = 1.5    # (best-median)/(median-min) must exceed this to trust
WORK_MAX_SIDE = 1500     # downscale for the angle search (speed; angle is scale-free)


def _binary(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    h, w = gray.shape
    if max(h, w) > WORK_MAX_SIDE:
        s = WORK_MAX_SIDE / max(h, w)
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (binary > 0).astype(np.float32)


def _profile_sharpness(rotated: np.ndarray) -> float:
    """Max over both axes of the projection's squared-difference energy."""
    best = 0.0
    for axis in (1, 0):  # row profile (yokogaki), then column profile (tategaki)
        profile = rotated.sum(axis=axis)
        diff = np.diff(profile)
        best = max(best, float(np.dot(diff, diff)))
    return best


def _rotate(binary: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = binary.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST,
                          borderValue=0)


def find_skew(img_bgr: np.ndarray) -> Deskew:
    binary = _binary(img_bgr)
    if binary.sum() < 50:  # essentially blank
        return Deskew(angle_deg=0.0, confidence=0.0)

    coarse = np.arange(-MAX_ANGLE, MAX_ANGLE + 1e-9, COARSE_STEP)
    scores = {a: _profile_sharpness(_rotate(binary, a)) for a in coarse}
    best_a = max(scores, key=scores.get)

    fine = np.arange(best_a - COARSE_STEP, best_a + COARSE_STEP + 1e-9, FINE_STEP)
    for a in fine:
        scores[round(float(a), 4)] = _profile_sharpness(_rotate(binary, a))
    best_a = max(scores, key=scores.get)

    values = np.array(list(scores.values()))
    median = float(np.median(values))
    spread = median - float(values.min())
    # baseline-invariant peakiness: how far the best angle stands above the
    # bulk, scaled by the spread (a constant offset from a figure cancels out)
    confidence = (scores[best_a] - median) / spread if spread > 0 else 0.0
    # ScanTailor convention: positive angle = clockwise skew to correct.
    return Deskew(angle_deg=round(float(best_a), 3), confidence=round(confidence, 3))


def is_confident(deskew: Deskew) -> bool:
    return deskew.confidence >= GOOD_CONFIDENCE
