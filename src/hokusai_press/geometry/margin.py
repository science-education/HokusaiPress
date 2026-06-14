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


EDGE_BAND_FRAC = 0.12        # outer 12% of each side is where scan shadows live
SHADOW_LINE_FRAC = 0.5       # an edge row/col >50% ink is a binding/ADF shadow


def strip_edge_shadows(binary: np.ndarray) -> np.ndarray:
    """Zero out near-edge rows/columns that are mostly ink.

    Book-binding / ADF scans leave a dark bar or band a few pixels inside an
    edge (not touching it, so the border-touching filter misses it). Such a bar
    is ~solid ink spanning the page, which otherwise blows the content box out to
    full-bleed and skews the deskew projection. A genuine text column is never
    >50% ink, so this only removes shadows.
    """
    out = binary.copy()
    h, w = out.shape
    col = (out > 0).mean(axis=0)
    row = (out > 0).mean(axis=1)
    ew, eh = int(w * EDGE_BAND_FRAC), int(h * EDGE_BAND_FRAC)
    for x in list(range(ew)) + list(range(w - ew, w)):
        if col[x] > SHADOW_LINE_FRAC:
            out[:, x] = 0
    for y in list(range(eh)) + list(range(h - eh, h)):
        if row[y] > SHADOW_LINE_FRAC:
            out[y, :] = 0
    return out


def _deskewed_binary(img_bgr: np.ndarray, deskew: Deskew) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    if abs(deskew.angle_deg) > 1e-3:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deskew.angle_deg, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderValue=255)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return strip_edge_shadows(binary)


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
    """Uniform-size, nombre-anchored margin normalization (sets margin.crop).

    The value ScanTailorAdvancedTATEGAKI adds: make every page the SAME physical
    size with the body in a consistent place, so the finished book is even.

    ADF-scanned books must have one uniform page size (different left/right page
    sizes are not a real book), so the output crop SIZE is taken as the largest
    content extent across the WHOLE document (in physical inches when dpi is
    known, so mixed-dpi sources still yield equal pages) plus the output margin.
    The max guarantees no page's body is ever clipped.

    Positioning fuses two strategies under a hard ">= output margin on all four
    sides" guarantee (the crop never touches the content):
    - when a page's nombre is CONFIDENT (a page number was assigned by the OCR
      sequence resolver), its vertical position is nombre-anchored to that
      parity's common offset so the body sits consistently across the book;
    - otherwise (low-confidence / no OCR), it falls back to the deterministic
      placement: horizontally centered with a constant head (top) margin.
    Either way the offset is clamped so every side keeps >= the output margin.
    """
    have = [p for p in pages if p.margin and p.margin.content]
    if not have:
        return
    use_dpi = all(p.dpi for p in have)

    def extent(p, px):  # content size in inches (if dpi) else pixels
        return px / p.dpi if use_dpi else px

    def confident(p):   # the OCR resolver only assigns a number it trusts
        return p.page_number is not None and p.margin.nombre_box is not None

    # ONE size for the whole document (uniform judgment size)
    max_w = max(extent(p, p.margin.content.width) for p in have)
    max_h = max(extent(p, p.margin.content.height) for p in have)

    # common nombre vertical offset per parity, from CONFIDENT pages only
    common_noff: dict[int, float] = {}
    by_parity: dict[int, list] = {0: [], 1: []}
    for p in have:
        by_parity[p.source.page_index % 2].append(p)
    for parity, group in by_parity.items():
        offs = [extent(p, p.margin.nombre_box.y0 - p.margin.content.y0)
                for p in group if confident(p)]
        if offs:
            common_noff[parity] = float(np.median(offs))

    for p in have:
        dpi = p.dpi if use_dpi else 1.0
        margin_px = output_margin_mm / 25.4 * (p.dpi if use_dpi else 96)
        crop_w = (max_w + 2 * (output_margin_mm / 25.4)) * (dpi if use_dpi else 1)
        crop_h = (max_h + 2 * (output_margin_mm / 25.4)) * (dpi if use_dpi else 1)
        c = p.margin.content
        x0 = c.x0 + c.width / 2 - crop_w / 2          # horizontal: centered
        noff = common_noff.get(p.source.page_index % 2)
        if confident(p) and noff is not None:
            # place the body so this page's nombre sits at the common offset
            y0 = p.margin.nombre_box.y0 - noff * (dpi if use_dpi else 1) - margin_px
        else:
            y0 = c.y0 - margin_px                      # fallback: constant head
        # clamp so EVERY side keeps >= the output margin (never touches content);
        # always feasible since the uniform crop >= content + 2*margin
        x0 = max(min(x0, c.x0 - margin_px), c.x1 + margin_px - crop_w)
        y0 = max(min(y0, c.y0 - margin_px), c.y1 + margin_px - crop_h)
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
