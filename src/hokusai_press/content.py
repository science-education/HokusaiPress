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

import os

import cv2
import numpy as np

from .model import Box, Flag, Region, RegionKind, SourceRef

PHOTO_AREA_FRAC = 0.01      # dense residual blob larger than this = photo
PHOTO_FILL = 0.2           # connected-component fill ratio separating photo from line art
TEXT_DILATE_PX = 6
LOW_COVERAGE_FRAC = 0.5    # text covers < 50% of ink -> flag for review

_ocr_engine = None


def _get_ocr_engine(model_dir: str, device: str, openvino_cache_dir=None):
    global _ocr_engine
    if _ocr_engine is None:
        from hybrid_ocr.pipeline import HybridOCR  # lazy, optional dependency

        # auto-resolve the model dir (the default "models" is relative and
        # usually absent from the cwd): fall back to hybrid-ocr's own resolver,
        # which finds the models bundled next to that package.
        if not os.path.isdir(model_dir):
            try:
                from hybrid_ocr.cli import resolve_model_dir
                model_dir = resolve_model_dir(None)
            except Exception:
                pass
        _ocr_engine = HybridOCR(model_dir=model_dir, device=device,
                                openvino_cache_dir=openvino_cache_dir)
    return _ocr_engine


def analyze(
    original_bgr: np.ndarray,
    ocr_bgr: np.ndarray,
    source: SourceRef,
    model_dir: str = "models",
    device: str = "auto",
    use_ocr: bool = True,
    layout_provider=None,
    openvino_cache_dir=None,
) -> tuple[list[Region], list[Flag]]:
    regions: list[Region] = []
    flags: list[Flag] = []
    inv_scale = 1.0 / max(source.ocr_scale, 1e-6)

    text_mask = None
    engine = None
    if use_ocr:
        try:
            engine = _get_ocr_engine(model_dir, device, openvino_cache_dir)
        except ImportError:
            # hybrid-ocr not installed: geometry-only is a supported mode
            use_ocr = False
        # NOTE: any other construction error (e.g. the requested device's
        # onnxruntime provider is missing, or models can't be found) is NOT
        # swallowed. OCR was explicitly requested, so failing loudly beats
        # silently degrading a whole batch to geometry-only.
    if engine is not None:
        try:
            result = engine(ocr_bgr)
        except Exception:
            result = None
            flags.append(Flag.OCR_FAILED)   # tolerate a single bad page
        if result is not None:
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

    # layout-model figure/photo regions (optional, e.g. RT-DETRv2)
    layout_boxes_px: list[tuple] = []
    if layout_provider is not None:
        try:
            for b in layout_provider.figures(ocr_bgr):
                x0, y0, x1, y1 = int(b.x0), int(b.y0), int(b.x1), int(b.y1)
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                layout_boxes_px.append((x0, y0, x1, y1))
                crop = ocr_bgr[max(0, y0):y1, max(0, x0):x1]
                kind = RegionKind.PHOTO if _is_continuous_tone(crop) else RegionKind.FIGURE
                regions.append(Region(
                    kind=kind,
                    box=Box(x0 * inv_scale, y0 * inv_scale,
                            x1 * inv_scale, y1 * inv_scale),
                    source="rtdetr",
                ))
        except Exception:
            pass

    # residual / photo detection on the OCR-resolution binary
    gray = cv2.cvtColor(ocr_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (binary > 0).astype(np.uint8)
    total_ink = int(ink.sum())

    residual = ink.copy()
    for (x0, y0, x1, y1) in layout_boxes_px:  # don't double-count layout figures
        residual[max(0, y0):y1, max(0, x0):x1] = 0
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


def _is_continuous_tone(crop: np.ndarray) -> bool:
    """True for halftone/photo regions (many distinct gray levels) vs line art
    (mostly two levels). Decides PHOTO vs FIGURE for a layout region."""
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    occupied = int((hist > gray.size * 0.01).sum())
    return occupied >= 6
