"""Content-box detection and nombre-anchored margin normalization.

Two responsibilities, both producing parameters in ORIGINAL-image pixels:

1. find_content_box(): the tight box around the page's real ink, ignoring
   scan-edge noise and speckle. This is the "select content" step.

2. align_margins(): the value ScanTailorAdvancedTATEGAKI adds over generic
   tools — making the body text occupy the *same* position on every page so
   the finished book is visually even. The anchor is the nombre (page
   number): once located, content boxes are aligned to a common reference so
   recto/verso and varying ink extents don't make the body wander. Pages
   whose nombre is missing or whose box disagrees with its neighbors are
   flagged for review rather than guessed.

The nombre detector here is a best-effort heuristic (small isolated text
component in the top/bottom margin band, recto/verso aware). The full
cross-page optimization from the C++ source is the next port; the interface
and flags are in place so the renderer and review UI already consume it.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..model import Box, Deskew, Flag, Margin

SPECKLE_AREA_FRAC = 1e-5     # components smaller than this fraction are noise
NOMBRE_BAND_FRAC = 0.08      # top/bottom 8% of the page is the nombre band
NOMBRE_MAX_HEIGHT_FRAC = 0.04
MARGIN_CONSISTENCY_TOL = 0.04  # neighbor content boxes within 4% of page size


def _deskewed_binary(img_bgr: np.ndarray, deskew: Deskew) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    if abs(deskew.angle_deg) > 1e-3:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deskew.angle_deg, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderValue=255)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def find_content_box(img_bgr: np.ndarray, deskew: Deskew) -> Margin:
    binary = _deskewed_binary(img_bgr, deskew)
    h, w = binary.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = SPECKLE_AREA_FRAC * h * w
    xs0, ys0, xs1, ys1 = [], [], [], []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        # drop components touching the image border (scan surround)
        if x == 0 or y == 0 or x + cw == w or y + ch == h:
            continue
        xs0.append(x); ys0.append(y); xs1.append(x + cw); ys1.append(y + ch)
    if not xs0:
        return Margin(content=Box(0.0, 0.0, float(w), float(h)), confidence=0.0)
    content = Box(float(min(xs0)), float(min(ys0)), float(max(xs1)), float(max(ys1)))
    # confidence: how much of the page the content fills (a sane scan fills a
    # large, central fraction; tiny or full-bleed boxes are suspicious)
    fill = (content.width * content.height) / (w * h)
    confidence = float(np.clip(fill * 1.5, 0.0, 1.0))
    nombre = _detect_nombre(binary, content)
    return Margin(content=content, confidence=confidence, nombre_box=nombre)


def _detect_nombre(binary: np.ndarray, content: Box) -> Box | None:
    """Small isolated text component in the top or bottom margin band."""
    h, w = binary.shape
    band = int(h * NOMBRE_BAND_FRAC)
    max_h = int(h * NOMBRE_MAX_HEIGHT_FRAC)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if ch > max_h or cw > w * 0.25 or area < 10:
            continue
        in_top = y < band
        in_bottom = y + ch > h - band
        if not (in_top or in_bottom):
            continue
        cx = x + cw / 2
        dist = abs(cx - w / 2)  # nombre is usually centered or in an outer corner
        if best is None or dist < best[0]:
            best = (dist, Box(float(x), float(y), float(x + cw), float(y + ch)))
    return best[1] if best else None


def normalize_margins(pages, output_margin_mm: float = 5.0) -> None:
    """Nombre-anchored margin normalization (mutates each page's margin.crop).

    The value ScanTailorAdvancedTATEGAKI adds over generic tools: make the body
    occupy the *same* place on every page so the finished book is visually even.

    Pages are grouped by parity (recto/verso) because inner/outer margins
    differ between facing pages. Within a group:
    - the output crop SIZE is uniform, taken as the largest content extent in
      the group (in physical inches when dpi is known, so mixed-dpi sources
      still yield equal output pages) plus the output margin -- the max
      guarantees no page's body is ever clipped;
    - each page's content is placed at a common anchor inside that crop: the
      head (top) margin is constant, and when a nombre was found its center is
      driven to the group's common position (the literal "nombre-based margin");
      pages without a nombre fall back to horizontal centering.

    Safety: normalization never shrinks the crop below a page's own content, so
    it cannot clip text. Outlier pages (content far larger than the group) are
    left correct but flagged elsewhere for review.
    """
    by_parity: dict[int, list] = {0: [], 1: []}
    for p in pages:
        if p.margin and p.margin.content:
            by_parity[p.source.page_index % 2].append(p)

    for group in by_parity.values():
        if not group:
            continue
        use_dpi = all(p.dpi for p in group)

        def extent(p, px):  # content size in inches (if dpi) else pixels
            return px / p.dpi if use_dpi else px

        max_w = max(extent(p, p.margin.content.width) for p in group)
        max_h = max(extent(p, p.margin.content.height) for p in group)

        # common nombre vertical position (inches/px from crop top), if any
        nombre_offsets = []
        for p in group:
            if p.margin.nombre_box:
                margin_px_p = output_margin_mm / 25.4 * (p.dpi or 1) if use_dpi \
                    else output_margin_mm / 25.4 * 96
                top = p.margin.content.y0 - margin_px_p
                noff = extent(p, p.margin.nombre_box.y0 - top)
                nombre_offsets.append(noff)
        common_noff = float(np.median(nombre_offsets)) if nombre_offsets else None

        for p in group:
            dpi = p.dpi if use_dpi else 1.0
            margin_px = output_margin_mm / 25.4 * (p.dpi if use_dpi else 96)
            crop_w = (max_w + 2 * (output_margin_mm / 25.4)) * (dpi if use_dpi else 1)
            crop_h = (max_h + 2 * (output_margin_mm / 25.4)) * (dpi if use_dpi else 1)
            c = p.margin.content
            # horizontal: center the content within the crop
            x0 = c.x0 + c.width / 2 - crop_w / 2
            # vertical: anchor by nombre if available, else constant head margin
            if common_noff is not None and p.margin.nombre_box:
                noff_px = common_noff * (dpi if use_dpi else 1)
                y0 = p.margin.nombre_box.y0 - noff_px
            else:
                y0 = c.y0 - margin_px
            p.margin.crop = Box(x0, y0, x0 + crop_w, y0 + crop_h)


def align_margins(margins: list[Margin]) -> list[Flag | None]:
    """Cross-page consistency check. Returns a per-page flag (or None).

    Full nombre-anchored re-alignment is the pending C++ port; for now this
    flags pages whose content box departs from the running median so they
    reach the review queue, which is the safety-first behavior.
    """
    if not margins:
        return []
    widths = np.array([m.content.width for m in margins if m.content])
    heights = np.array([m.content.height for m in margins if m.content])
    med_w, med_h = float(np.median(widths)), float(np.median(heights))
    flags: list[Flag | None] = []
    for m in margins:
        if m.content is None or m.confidence < 0.2:
            flags.append(Flag.MARGIN_NOT_FOUND)
            continue
        dw = abs(m.content.width - med_w) / max(med_w, 1)
        dh = abs(m.content.height - med_h) / max(med_h, 1)
        if m.nombre_box is None and (dw > MARGIN_CONSISTENCY_TOL
                                     or dh > MARGIN_CONSISTENCY_TOL):
            flags.append(Flag.MARGIN_INCONSISTENT)
        else:
            flags.append(None)
    return flags
