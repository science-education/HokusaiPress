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
from .ocr.base import FIGURE_LIKE_LABELS

PHOTO_AREA_FRAC = 0.01      # dense residual blob larger than this = photo
PHOTO_FILL = 0.2           # connected-component fill ratio separating photo from line art
TEXT_DILATE_PX = 6
LOW_COVERAGE_FRAC = 0.3    # text+figure covers < 30% of ink -> flag for review.
# Calibrated on real books: detected-line polygons cover only ~40% of text ink
# even on clean dense pages (boxes are tight), so 0.5 flagged the median page.
# At 0.3 only genuine outliers (OCR truly missed most text) reach the queue.
TEXT_LAYOUT_LABELS = frozenset({"page_number", "running_head"})
FIGURE_LAYOUT_LABELS = FIGURE_LIKE_LABELS | frozenset({"image"})

def analyze(
    original_bgr: np.ndarray,
    ocr_bgr: np.ndarray,
    source: SourceRef,
    model_dir: str = "models",
    device: str = "auto",
    ocr_engine: str = "hybrid",
    layout_engine: str | None = None,
    text_engine: str | None = None,
    use_ocr: bool = True,
    layout_provider=None,
    openvino_cache_dir=None,
    runtime: str | None = "paddle",
) -> tuple[list[Region], list[Flag]]:
    from .geometry.margin import remove_edge_shadows

    regions: list[Region] = []
    flags: list[Flag] = []
    inv_scale = 1.0 / max(source.ocr_scale, 1e-6)
    # drop binding/ADF shadows up front so OCR can't read a bar as a text line
    # and the residual-photo pass can't mistake one for a figure
    ocr_bgr = remove_edge_shadows(ocr_bgr)

    text_mask = None
    engine = None
    layout_boxes_px: list[tuple] = []
    if use_ocr:
        try:
            from .ocr import get_ocr_engine

            engine = get_ocr_engine(
                ocr_engine,
                model_dir,
                device,
                openvino_cache_dir,
                runtime,
                layout_engine=layout_engine,
                text_engine=text_engine,
            )
        except ImportError:
            # optional OCR engine not installed: geometry-only is a supported mode
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
            lines = result.get("lines", [])
            text_layout_boxes = []
            for lb in result.get("layout_boxes", []):
                label = lb.get("label")
                if label in TEXT_LAYOUT_LABELS:
                    text_layout_boxes.append((label, tuple(float(v) for v in lb["box"])))
            for ln in lines:
                x0, y0, x1, y1 = ln["box"]
                poly = ln.get("polygon")
                if poly is None:
                    poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                poly = np.array(poly, dtype=np.int32)
                cv2.fillPoly(text_mask, [poly], 1)
                layout_label = _matching_layout_label(
                    (float(x0), float(y0), float(x1), float(y1)),
                    text_layout_boxes,
                )
                regions.append(
                    Region(
                        kind=RegionKind.TEXT,
                        box=Box(x0 * inv_scale, y0 * inv_scale,
                                x1 * inv_scale, y1 * inv_scale),
                        source=ln.get("source", "dbnet"),
                        ocr_text=ln.get("text") or None,
                        ocr_conf=ln.get("det_score"),
                        layout_label=layout_label,
                    )
                )
            if not lines:
                flags.append(Flag.NO_TEXT)
            # Furniture (folio / running head) is recognized by the engine but is not
            # part of the reading-order lines. Surface it as a tagged TEXT region so
            # nombre can read the page number.
            for lb in result.get("layout_boxes", []):
                label = lb.get("label")
                txt = lb.get("text")
                if label not in TEXT_LAYOUT_LABELS or not txt:
                    continue
                fx0, fy0, fx1, fy1 = lb["box"]
                regions.append(Region(
                    kind=RegionKind.TEXT,
                    box=Box(fx0 * inv_scale, fy0 * inv_scale,
                            fx1 * inv_scale, fy1 * inv_scale),
                    source=lb.get("source", "deim"),
                    ocr_text=txt,
                    layout_label=label,
                ))
            for lb in result.get("layout_boxes", []):
                # Engines now emit ALL classified regions (text/page_number/figure/...).
                # Only figure-like classes become image regions; an absent label keeps
                # legacy behaviour (treated as figure).
                label = lb.get("label")
                if label is not None and label not in FIGURE_LAYOUT_LABELS:
                    continue
                x0, y0, x1, y1 = [int(v) for v in lb["box"]]
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                layout_boxes_px.append((x0, y0, x1, y1))
                crop = ocr_bgr[max(0, y0):y1, max(0, x0):x1]
                kind = RegionKind.PHOTO if _is_continuous_tone(crop) else RegionKind.FIGURE
                regions.append(Region(
                    kind=kind,
                    box=Box(x0 * inv_scale, y0 * inv_scale,
                            x1 * inv_scale, y1 * inv_scale),
                    source=lb.get("source", ocr_engine),
                ))

    # layout-model figure/photo regions (optional, e.g. RT-DETRv2)
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

    # residual / photo detection on a clamp-thresholded ink mask (not raw Otsu):
    # Otsu turns faint show-through into "ink" and so invents a spurious photo on
    # a blank page; the absolute-floored threshold rejects it.
    from .geometry.margin import ink_threshold
    gray = cv2.cvtColor(ocr_bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray <= ink_threshold(gray)).astype(np.uint8)
    total_ink = int(ink.sum())

    residual = ink.copy()
    for (x0, y0, x1, y1) in layout_boxes_px:  # don't double-count layout figures
        residual[max(0, y0):y1, max(0, x0):x1] = 0
    has_text = text_mask is not None and any(r.kind == RegionKind.TEXT for r in regions)
    if text_mask is not None:
        dil = cv2.dilate(text_mask, np.ones((TEXT_DILATE_PX * 2 + 1,) * 2, np.uint8))
        residual[dil > 0] = 0

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
            residual[y:y + ch, x:x + cw] = 0   # this ink is a figure, not lost text

    # low-coverage flag AFTER figure/photo removal: only ink that is neither text
    # nor figure/photo counts as "missing text". An illustration-heavy page (lots
    # of figure ink) no longer false-flags -- it's not lost OCR.
    if has_text and total_ink > 0:
        covered = total_ink - int(residual.sum())
        if covered / total_ink < LOW_COVERAGE_FRAC:
            flags.append(Flag.OCR_LOW_COVERAGE)
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


def _matching_layout_label(line_box, layout_boxes) -> str | None:
    for label, box in layout_boxes:
        if _box_center_inside(line_box, box) or _box_iou(line_box, box) >= 0.3:
            return label
    return None


def _box_center_inside(inner, outer) -> bool:
    x0, y0, x1, y1 = inner
    ox0, oy0, ox1, oy1 = outer
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return ox0 <= cx <= ox1 and oy0 <= cy <= oy1


def _box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0
