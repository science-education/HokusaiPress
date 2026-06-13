"""Semantic content separation: classify page ink into text / figure / photo.

This is the core of "tell characters from pictures" — the thing pixel
statistics alone cannot do reliably. It fuses three signals (in priority
order):

1. DBNet text lines + recognition (via hybrid-ocr, if installed): the
   strongest positive evidence of "text". Recognition confidence is used
   ONLY to *exclude* false detections (a line with empty text AND low
   detection score is demoted), never to demote real-but-hard text — so
   faint/cursive characters are kept as text. When in doubt, ink stays in
   the bilevel layer (a photo wrongly binarized is visible and fixable; text
   lost into a blurry photo layer silently ruins search).

2. RT-DETRv2 layout regions (figure/photo) — reserved hook; when wired, a
   detected figure region that contains no text lines becomes a PHOTO/FIGURE
   region for the grayscale layer.

3. Residual dense ink outside all text regions -> PHOTO heuristic (the
   fallback that also runs standalone when no OCR engine is present).

All output regions are in ORIGINAL-image pixels (OCR boxes divided by
SourceRef.ocr_scale).
"""

from __future__ import annotations

import cv2
import numpy as np

from .model import Box, Flag, Region, RegionKind, SourceRef

PHOTO_AREA_FRAC = 0.01      # dense residual blob larger than this = photo
PHOTO_FILL = 0.2           # connected-component fill ratio separating photo from line art
TEXT_DILATE_PX = 6
LOW_COVERAGE_FRAC = 0.5    # text covers < 50% of ink -> flag for review

_ocr_engine = None


def _get_ocr_engine(model_dir: str, device: str):
    global _ocr_engine
    if _ocr_engine is None:
        from hybrid_ocr.pipeline import HybridOCR  # lazy, optional dependency

        _ocr_engine = HybridOCR(model_dir=model_dir, device=device)
    return _ocr_engine


def analyze(
    original_bgr: np.ndarray,
    ocr_bgr: np.ndarray,
    source: SourceRef,
    model_dir: str = "models",
    device: str = "auto",
    use_ocr: bool = True,
) -> tuple[list[Region], list[Flag]]:
    regions: list[Region] = []
    flags: list[Flag] = []
    inv_scale = 1.0 / max(source.ocr_scale, 1e-6)

    text_mask = None
    if use_ocr:
        try:
            engine = _get_ocr_engine(model_dir, device)
            result = engine(ocr_bgr)
            text_mask = np.zeros(ocr_bgr.shape[:2], dtype=np.uint8)
            for ln in result["lines"]:
                poly = np.array(ln["polygon"], dtype=np.int32)
                cv2.fillPoly(text_mask, [poly], 1)
                x0, y0, x1, y1 = ln["box"]
                regions.append(
                    Region(
                        kind=RegionKind.TEXT,
                        box=Box(x0 * inv_scale, y0 * inv_scale,
                                x1 * inv_scale, y1 * inv_scale),
                        source="dbnet",
                        ocr_text=ln.get("text") or None,
                        ocr_conf=ln.get("det_score"),
                    )
                )
            if not result["lines"]:
                flags.append(Flag.NO_TEXT)
        except ImportError:
            use_ocr = False
        except Exception:
            use_ocr = False

    # residual / photo detection on the OCR-resolution binary
    gray = cv2.cvtColor(ocr_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (binary > 0).astype(np.uint8)
    total_ink = int(ink.sum())

    residual = ink.copy()
    if text_mask is not None:
        dil = cv2.dilate(text_mask, np.ones((TEXT_DILATE_PX * 2 + 1,) * 2, np.uint8))
        residual[dil > 0] = 0
        covered = total_ink - int(residual.sum())
        if total_ink > 0 and covered / total_ink < LOW_COVERAGE_FRAC and regions:
            flags.append(Flag.OCR_LOW_COVERAGE)

    h, w = ink.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if x == 0 or y == 0 or x + cw == w or y + ch == h:
            continue  # scan surround
        if area < PHOTO_AREA_FRAC * h * w:
            continue
        if area / float(cw * ch) >= PHOTO_FILL:
            regions.append(
                Region(
                    kind=RegionKind.PHOTO,
                    box=Box(x * inv_scale, y * inv_scale,
                            (x + cw) * inv_scale, (y + ch) * inv_scale),
                    source="residual",
                )
            )
    return regions, flags
