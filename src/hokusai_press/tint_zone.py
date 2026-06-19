"""High-quality tint-zone analysis and overlay preparation.

A "tint zone" is a region of the page that carries a flat or slowly-varying
grey background (平網 / halftone tint) -- usually a header, footer, or article
background panel -- with text or other marks printed on top of it.

The goal is to reproduce this appearance in the output PDF with maximum fidelity:
  - The tint background must be rendered at its actual per-pixel colour.
  - Text strokes on the tint must remain sharp bilevel.
  - White paper areas within the panel boundary must stay white.
  - Non-rectangular shapes (rounded corners, circular badges extending beyond
    the rectangular band, L-shapes, etc.) must be clipped precisely.

Architecture (layered compositing with Multiply blend):

  1. Bilevel base layer  -- Otsu-binarised.  Within the tint shape the
                           background is erased to white; text strokes are
                           identified with a zone-local threshold and kept black.
  2. Tint overlay        -- Greyscale JPEG image placed with /BM /Multiply,
                           clipped to the exact tint-shape contour via a PDF
                           clip path (W n operator).
                           • tint background  → original grey value × 1.0 = tint
                           • paper (> TINT_HI) → 255 × 1.0 = white (identity)
                           • text stroke area → any × 0 = 0 (stays black)

Contour-based clipping (replacing the earlier rectangular approach):
  1. Compute tint_mask = pixels in [TINT_LO, TINT_HI] on the full grey image.
  2. Apply morphological CLOSE to fill text-stroke holes and connect any
     extensions (e.g. a circular badge hanging below the main band).
     The close radius is ~3 % of the shorter page dimension.
  3. cv2.findContours(RETR_EXTERNAL) gives the outer boundary of each
     connected tint region.
  4. cv2.approxPolyDP simplifies each contour to a manageable polyline;
     the polyline is emitted as PDF "m / l / h" operators before "W n"
     to set a clip path for the Multiply overlay.
  5. The bounding rect of the contour (which may be larger than the
     detect_tint_panels hint rect) is used as the JPEG image boundary.

Overlap deduplication:
  Adjacent panels that share tint pixels (e.g. a horizontal band meeting a
  vertical binding-shadow column at a corner) are merged into one connected
  component by MORPH_CLOSE and share one contour.  A second Multiply pass
  over the same area is therefore impossible by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

TINT_LO: int = 60
TINT_HI: int = 220

# A bounding rect with tint coverage >= this fraction is called "rect".
RECT_COVERAGE_MIN: float = 0.95
# Below this coverage, shape may be "rounded_rect" or "poly".
ROUNDED_COVERAGE_MIN: float = 0.85
# Minimum corner-curve pixels to qualify as a rounded rect (fraction of short side).
CORNER_RADIUS_MIN_FRAC: float = 0.02

# MORPH_CLOSE kernel: fraction of the shorter dimension of the LOCAL crop.
# Applied per-panel on an expanded crop, so fills text-stroke holes without
# bridging distant separate panels (which full-page MORPH_CLOSE would do).
_CLOSE_FRAC: float = 0.030
_CLOSE_ITERATIONS: int = 2

# Search margin beyond each panel hint rect: fraction of the larger panel dim.
# Generous enough to capture circular badges / rounded corners that hang just
# outside the detection hint while staying well within the panel neighbourhood.
_EXPAND_FRAC: float = 0.10

# Contour simplification: fraction of arc length used as epsilon for
# cv2.approxPolyDP.  Keeps ~0.3 % of perimeter as maximum deviation,
# but capped at _MAX_APPROX_ERR_PX so curved features on large contours
# (e.g. a circle badge attached to a wide header band) retain enough vertices.
_APPROX_FRAC: float = 0.003
_MAX_APPROX_ERR_PX: float = 3.0   # hard cap on polygon approximation error (px)

# Iterative re-expansion: if the local contour touches the crop boundary we
# may have clipped the shape.  Re-expand up to _MAX_EXPAND_ITERS times,
# growing the margin by _EXPAND_GROW each time.
_MAX_EXPAND_ITERS: int = 3
_EXPAND_GROW: float = 2.0

# Circle detection: if a contour's compactness (4π·area/perimeter²) exceeds
# this threshold the shape is treated as a circle and rendered with Bézier
# arcs rather than a polygon (exact to < 0.03 % error).
_CIRCULARITY_MIN: float = 0.85

# Seam margin between a clipped column and an adjacent row zone (fraction of
# page height).  Keeps the column short of the row boundary so a transition
# curve at the seam (e.g. column flaring into a header/footer) lands entirely
# inside the row zone's expanded crop rather than being split between zones.
_SEAM_MARGIN_FRAC: float = 0.025


@dataclass
class TintZone:
    """Rendering data for one tint panel."""

    rect: tuple[int, int, int, int]      # (x0, y0, x1, y1) in page-image pixels
    overlay_img: np.ndarray              # HxW uint8 greyscale; 255 = transparent
    page_contour: np.ndarray | None = None  # (N,1,2) int32, page-image space (y-down)
    shape_type: str = "rect"             # "rect" | "rounded_rect" | "poly" | "circle"
    corner_radius: float = 0.0
    contour_pts: list = field(default_factory=list)
    circle_fit: tuple[float, float, float] | None = None  # (cx, cy, r) page-image coords y-down


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tint_zones(
    gray: np.ndarray,
    panels: list[dict],
) -> list[TintZone]:
    """Convert raw detected panel dicts to TintZone objects.

    ``panels`` is the output of ``tint_panel.detect_tint_panels``.

    Strategy:
      1. Split panels by orientation (row vs column) and clip column panels
         against row-panel y-intervals to prevent overlap / double-blending.
      2. For each panel, detect its accurate shape contour using a LOCAL
         MORPH_CLOSE on an orientation-aware expanded crop:
           - Row panels: expand downward only (captures circular badges that
             hang below the detection hint rect).
           - Column panels: expand horizontally only (captures shape edges).
         Orientation-specific expansion avoids including tint from the
         intersecting band, which would create a C-shape contour.
      3. Build a TintZone with the local contour translated to page coordinates.

    Returns a list of non-overlapping TintZone objects.
    """
    if not panels:
        return []

    h, w = gray.shape[:2]

    # Step 1: separate by orientation and clip columns against rows.
    # A small seam margin keeps the column short of each row boundary so that
    # a column-to-row transition curve (e.g. a decorative strip flaring into a
    # footer) is left entirely inside the row zone's expanded crop instead of
    # being cut in half by an arbitrary y-split between the two zones.
    row_panels, col_panels = _split_by_orientation(panels, w, h)
    row_y_intervals = [(p["rect"][1], p["rect"][3]) for p in row_panels]
    seam_margin = max(15, int(h * _SEAM_MARGIN_FRAC))

    clipped_cols: list[dict] = []
    for cp in col_panels:
        for rect in _clip_col_panel(cp["rect"], row_y_intervals, h, seam_margin):
            clipped_cols.append({"rect": rect, "gray": cp.get("gray", 0.67)})

    # Build column zones FIRST so their claimed area is known before any row
    # zone's contour search runs.  A row zone's iterative downward/upward
    # growth (added to capture badges and seam curves) can otherwise re-trace
    # tint pixels that a column zone already owns -- both zones would then
    # paint the same pixels with their own Multiply overlay, doubling the
    # tint there (visible as an abnormally dark band down the column).
    col_zones: list[TintZone] = []
    for p in sorted(clipped_cols, key=lambda p: -(p["rect"][2]-p["rect"][0])*(p["rect"][3]-p["rect"][1])):
        z = _build_zone_with_local_contour(gray, p, w, h)
        if z is not None:
            col_zones.append(z)

    exclude_mask = np.zeros((h, w), dtype=bool)
    for z in col_zones:
        ex0, ey0, ex1, ey1 = z.rect
        if z.page_contour is not None:
            sub = np.zeros((ey1 - ey0, ex1 - ex0), dtype=np.uint8)
            cnt = z.page_contour.copy()
            cnt[:, 0, 0] -= ex0
            cnt[:, 0, 1] -= ey0
            cv2.fillPoly(sub, [cnt], 255)
            exclude_mask[ey0:ey1, ex0:ex1] |= sub > 0
        else:
            exclude_mask[ey0:ey1, ex0:ex1] = True

    # Row zones search a version of the page with already-claimed column
    # pixels hidden (forced to "paper"), so their contour can never grow back
    # into column territory.  Their final rect/contour therefore naturally
    # excludes it; the PDF clip path (built from that contour) is what
    # actually gates painting, so this has no effect on legitimate areas.
    search_gray = gray.copy()
    search_gray[exclude_mask] = 255

    row_zones: list[TintZone] = []
    for p in sorted(row_panels, key=lambda p: -(p["rect"][2]-p["rect"][0])*(p["rect"][3]-p["rect"][1])):
        z = _build_zone_with_local_contour(search_gray, p, w, h)
        if z is not None:
            row_zones.append(z)

    zones = row_zones + col_zones

    return zones


def _build_zone_with_local_contour(
    gray: np.ndarray,
    panel: dict,
    page_w: int,
    page_h: int,
) -> "TintZone | None":
    """Find the tint-shape contour for one panel using a LOCAL MORPH_CLOSE crop.

    Expansion is orientation-aware to avoid including adjacent panels' tint:
    row panels expand only downward (captures badges below the hint rect),
    column panels expand only horizontally (captures side-edge variations).

    Iterative re-expansion: if the found contour touches the expansion boundary
    the badge may extend further out; the crop is doubled up to _MAX_EXPAND_ITERS
    times in the allowed direction so the full shape is captured.

    Polygon precision: epsilon is capped at _MAX_APPROX_ERR_PX regardless of
    total perimeter, so curved features on large contours (e.g. a circle badge
    attached to a wide header) get enough polygon vertices to look smooth.

    Circle detection: contours with compactness ≥ _CIRCULARITY_MIN are stored
    with a circle_fit (cx, cy, r) so the PDF clip path can use exact Bézier arcs.
    """
    x0, y0, x1, y1 = panel["rect"]
    pw, ph = max(1, x1 - x0), max(1, y1 - y0)
    ratio = pw / ph

    if ratio >= 2.0:          # landscape / row → extend downward for badges
        exp_left = exp_right = 0
        # Small FIXED (non-iterating) upward margin, sized to match the seam
        # buffer that _clip_col_panel leaves around row boundaries: large
        # enough to capture a rounded-corner / transition curve that pokes
        # above the hint rect at a zone seam (e.g. a column flaring into a
        # footer), but deliberately not grown iteratively -- the column tint
        # runs the full page height right at this x-range, so any large or
        # iterative upward growth would bridge into it and recreate the
        # C-shape merge bug.
        exp_up = max(15, int(page_h * _SEAM_MARGIN_FRAC))
        exp_down = max(20, int(ph * _EXPAND_FRAC))
    elif ratio <= 0.5:        # portrait / column → extend horizontally only
        exp_left = exp_right = max(20, int(pw * _EXPAND_FRAC))
        exp_up = exp_down = 0
    else:                      # square-ish → all-round expansion
        margin = max(15, int(min(pw, ph) * _EXPAND_FRAC))
        exp_left = exp_right = exp_up = exp_down = margin

    best_cnt = None
    ex0 = ey0 = ex1 = ey1 = 0

    for _attempt in range(_MAX_EXPAND_ITERS):
        ex0 = max(0, x0 - exp_left);    ey0 = max(0, y0 - exp_up)
        ex1 = min(page_w, x1 + exp_right); ey1 = min(page_h, y1 + exp_down)

        local_gray = gray[ey0:ey1, ex0:ex1]
        local_contours = _find_tint_contours(local_gray)

        lx = (x0 + x1) // 2 - ex0
        ly = (y0 + y1) // 2 - ey0

        best_cnt = None
        for cnt in local_contours:
            if cv2.pointPolygonTest(cnt, (float(lx), float(ly)), False) >= 0:
                best_cnt = cnt
                break

        if best_cnt is None:
            break  # no contour → use rect fallback below

        # Check whether the contour touches the expansion boundary in the
        # direction we are allowed to grow; if so, double the margin and retry.
        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        crop_h, crop_w = local_gray.shape[:2]
        need_more = False
        if ratio >= 2.0 and (by + bh) >= crop_h - 2 and ey1 < page_h:
            exp_down = min(page_h - y1, int(exp_down * _EXPAND_GROW))
            need_more = True
        elif ratio <= 0.5:
            if bx <= 2 and ex0 > 0:
                exp_left = min(x0, int(exp_left * _EXPAND_GROW))
                need_more = True
            if (bx + bw) >= crop_w - 2 and ex1 < page_w:
                exp_right = min(page_w - x1, int(exp_right * _EXPAND_GROW))
                need_more = True

        if not need_more:
            break

    if best_cnt is None:
        return _make_tint_zone_rect(gray, panel["rect"])

    # Pad the contour by a couple of pixels before simplifying.  At a tangent
    # point where a rounded corner meets a flat edge, approxPolyDP's vertex
    # placement can leave a hairline notch (a few px) between the polygon and
    # the true tint boundary; that notch shows up as a black wedge in the
    # bilevel layer.  A small dilation removes it at the cost of a few extra
    # paper pixels being marked in-shape, which Multiply renders as identity.
    pad_mask = np.zeros(local_gray.shape[:2], dtype=np.uint8)
    cv2.drawContours(pad_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
    pad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pad_mask = cv2.dilate(pad_mask, pad_kernel, iterations=1)
    padded_contours, _ = cv2.findContours(
        pad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if padded_contours:
        best_cnt = max(padded_contours, key=cv2.contourArea)

    # Translate contour from local crop → page coordinates.
    page_cnt = best_cnt.copy()
    page_cnt[:, 0, 0] += ex0
    page_cnt[:, 0, 1] += ey0

    bx, by, bw, bh = cv2.boundingRect(page_cnt)
    bx = max(0, bx);  by = max(0, by)
    bw = min(bw, page_w - bx);  bh = min(bh, page_h - by)
    contour_rect = (bx, by, bx + bw, by + bh)

    # Adaptive approxPolyDP: cap error at _MAX_APPROX_ERR_PX so that curved
    # sections on large contours get adequate vertex density.
    peri = cv2.arcLength(best_cnt, True)
    epsilon = max(1.5, min(_MAX_APPROX_ERR_PX, peri * _APPROX_FRAC))
    approx = cv2.approxPolyDP(best_cnt, epsilon, True)
    approx_page = approx.copy()
    approx_page[:, 0, 0] += ex0
    approx_page[:, 0, 1] += ey0

    # Circle fit: if the contour is nearly circular use a parametric fit so the
    # PDF clip can use exact Bézier arcs instead of a polygon.
    circle_fit: tuple[float, float, float] | None = None
    if _contour_circularity(best_cnt) >= _CIRCULARITY_MIN:
        (cx_l, cy_l), r_l = cv2.minEnclosingCircle(best_cnt)
        circle_fit = (float(cx_l) + ex0, float(cy_l) + ey0, float(r_l))

    return _make_tint_zone_with_contour(gray, contour_rect, approx_page, circle_fit)


def make_tint_overlay(gray_crop: np.ndarray) -> np.ndarray:
    """Return the overlay image for a tint panel crop.

    Paper pixels (> TINT_HI) are set to 255 so Multiply has no effect there.
    All other pixels keep their original grey value.
    """
    overlay = gray_crop.copy()
    overlay[gray_crop > TINT_HI] = 255
    return overlay


def contour_to_pdf_path(contour: np.ndarray, page_h_px: int) -> bytes:
    """Convert an (N,1,2) OpenCV contour to PDF path bytes.

    Coordinate conversion: image space (y-down) → PDF space (y-up).
      pdf_y = page_h_px - image_y

    Returns bytes suitable for insertion before "W n" in a PDF content stream.
    The path uses the m / l / h operators (moveto / lineto / closepath).
    All coordinates are integer pixel values; the caller's responsibility to
    apply any further scale transform (e.g. the 72/dpi scale set by
    _set_physical_page_size wraps the whole page in a cm, so nothing extra
    is needed here).
    """
    if contour is None or len(contour) < 3:
        return b""
    parts: list[str] = []
    for i, pt in enumerate(contour):
        x = int(pt[0][0])
        y = page_h_px - int(pt[0][1])   # y-up
        if i == 0:
            parts.append(f"{x} {y} m")
        else:
            parts.append(f"{x} {y} l")
    parts.append("h")
    return (" ".join(parts) + "\n").encode()


def classify_tint_shape(
    tint_mask: np.ndarray,
) -> tuple[str, float, list]:
    """Classify the shape of a tint panel from its binary mask.

    Returns (shape_type, corner_radius_px, contour_pts).
    """
    h, w = tint_mask.shape[:2]
    coverage = float(tint_mask.mean())

    if coverage >= RECT_COVERAGE_MIN:
        return "rect", 0.0, []

    contours, _ = cv2.findContours(
        tint_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return "rect", 0.0, []
    cnt = max(contours, key=cv2.contourArea)

    if coverage < ROUNDED_COVERAGE_MIN:
        pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
        return "poly", 0.0, pts

    r = _estimate_corner_radius(tint_mask, h, w)
    min_r = max(1.0, min(h, w) * CORNER_RADIUS_MIN_FRAC)
    if r >= min_r:
        return "rounded_rect", r, []

    return "rect", 0.0, []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_tint_contours(gray: np.ndarray) -> list[np.ndarray]:
    """Compute tint region outer contours on the full grey image.

    MORPH_CLOSE fills text-stroke holes within tint panels so that the
    contour traces the panel boundary correctly (including extensions like
    circular badges).  Connected tint regions (e.g. a horizontal band whose
    binding-shadow column is also tint-coloured) are merged into one contour.
    """
    h, w = gray.shape[:2]
    tint_mask = ((gray >= TINT_LO) & (gray <= TINT_HI)).astype(np.uint8)

    close_px = max(15, int(min(h, w) * _CLOSE_FRAC))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
    )
    closed = cv2.morphologyEx(
        tint_mask, cv2.MORPH_CLOSE, kernel, iterations=_CLOSE_ITERATIONS
    )

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return list(contours)


def _make_tint_zone_rect(
    gray: np.ndarray,
    rect: tuple[int, int, int, int],
) -> TintZone | None:
    """Fallback: build a rectangular TintZone with no contour."""
    x0, y0, x1, y1 = rect
    h_full, w_full = gray.shape[:2]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w_full, x1); y1 = min(h_full, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    overlay = make_tint_overlay(gray[y0:y1, x0:x1])
    tint_mask = (
        (gray[y0:y1, x0:x1] >= TINT_LO) & (gray[y0:y1, x0:x1] <= TINT_HI)
    ).astype(np.uint8)
    shape_type, corner_radius, contour_pts = classify_tint_shape(tint_mask)
    return TintZone(
        rect=(x0, y0, x1, y1),
        overlay_img=overlay,
        page_contour=None,
        shape_type=shape_type,
        corner_radius=corner_radius,
        contour_pts=contour_pts,
    )


def _make_tint_zone_with_contour(
    gray: np.ndarray,
    rect: tuple[int, int, int, int],
    contour: np.ndarray,
    circle_fit: tuple[float, float, float] | None = None,
) -> TintZone | None:
    """Build a TintZone using the actual contour (not just bounding rect)."""
    x0, y0, x1, y1 = rect
    h_full, w_full = gray.shape[:2]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w_full, x1); y1 = min(h_full, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    overlay = make_tint_overlay(gray[y0:y1, x0:x1])

    if circle_fit is not None:
        shape_type = "circle"
        corner_radius = 0.0
    else:
        gray_crop = gray[y0:y1, x0:x1]
        tint_mask = (
            (gray_crop >= TINT_LO) & (gray_crop <= TINT_HI)
        ).astype(np.uint8)
        shape_type, corner_radius, _ = classify_tint_shape(tint_mask)
        if len(contour) > 8:
            shape_type = "poly"

    return TintZone(
        rect=(x0, y0, x1, y1),
        overlay_img=overlay,
        page_contour=contour,
        shape_type=shape_type,
        corner_radius=corner_radius,
        contour_pts=[],
        circle_fit=circle_fit,
    )


# ---------------------------------------------------------------------------
# Orientation split and column-clip helpers (used by build_tint_zones)
# ---------------------------------------------------------------------------

def _split_by_orientation(
    panels: list[dict],
    page_w: int,
    page_h: int,
) -> tuple[list[dict], list[dict]]:
    row_panels: list[dict] = []
    col_panels: list[dict] = []
    for p in panels:
        x0, y0, x1, y1 = p["rect"]
        pw = x1 - x0; ph = y1 - y0
        if pw == 0 or ph == 0:
            continue
        ratio = pw / ph
        if ratio >= 2.0:
            row_panels.append(p)
        elif ratio <= 0.5:
            col_panels.append(p)
        else:
            row_panels.append(p)
    return row_panels, col_panels


def _clip_col_panel(
    col_rect: tuple[int, int, int, int],
    row_y_intervals: list[tuple[int, int]],
    page_h: int,
    seam_margin: int = 0,
) -> list[tuple[int, int, int, int]]:
    """Clip a column rect against row y-intervals.

    ``seam_margin`` (page pixels) is subtracted from each free interval's edge
    that borders a row interval, so the column stops short of the row
    boundary.  This leaves a buffer strip that belongs to neither zone's
    *hint rect*, which the row zone's small upward expansion (see
    ``_build_zone_with_local_contour``) can then claim -- keeping a
    column-to-row transition curve whole instead of split at an arbitrary
    y-coordinate.  Default 0 preserves exact-boundary clipping for callers
    that don't need the buffer (e.g. unit tests).
    """
    x0, y0, y1_orig = col_rect[0], col_rect[1], col_rect[3]
    x1 = col_rect[2]
    blocked: list[tuple[int, int]] = []
    for ry0, ry1 in row_y_intervals:
        lo = max(y0, ry0); hi = min(y1_orig, ry1)
        if hi > lo:
            blocked.append((lo, hi))
    if not blocked:
        return [(x0, y0, x1, y1_orig)]
    blocked.sort()
    merged: list[tuple[int, int]] = [blocked[0]]
    for lo, hi in blocked[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    free: list[tuple[int, int, int, int]] = []
    cursor = y0
    for blo, bhi in merged:
        if cursor < blo:
            seg_end = max(cursor, blo - seam_margin)
            if seg_end > cursor:
                free.append((x0, cursor, x1, seg_end))
        cursor = max(cursor, bhi + seam_margin)
    if cursor < y1_orig:
        free.append((x0, cursor, x1, y1_orig))
    min_h = max(4, int(page_h * 0.030))
    return [(x0, y0c, x1, y1c) for (x0, y0c, x1, y1c) in free if y1c - y0c >= min_h]


# ---------------------------------------------------------------------------
# Contour geometry helpers
# ---------------------------------------------------------------------------

def _contour_circularity(cnt: np.ndarray) -> float:
    """4π·area / perimeter²  (1.0 for a perfect circle, ≤ 1 for everything else)."""
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri < 1.0:
        return 0.0
    return 4.0 * np.pi * area / (peri * peri)


def circle_to_pdf_path(cx: float, cy: float, r: float, page_h_px: int) -> bytes:
    """Exact-circle PDF clip path using 4 cubic Bézier arcs (error < 0.03 %).

    Coordinate conversion: image space (y-down) → PDF space (y-up).
    The Bézier control-point factor k = 4(√2-1)/3 ≈ 0.5523 gives the best
    rational approximation to a circle with cubic polynomials.
    """
    pcx = cx
    pcy = float(page_h_px) - cy   # y-flip
    k = 0.5522847498 * r
    # 4 arcs: right → top → left → bottom (PDF y-up, counterclockwise)
    parts = [f"{pcx+r:.2f} {pcy:.2f} m"]
    parts.append(f"{pcx+r:.2f} {pcy+k:.2f} {pcx+k:.2f} {pcy+r:.2f} {pcx:.2f} {pcy+r:.2f} c")
    parts.append(f"{pcx-k:.2f} {pcy+r:.2f} {pcx-r:.2f} {pcy+k:.2f} {pcx-r:.2f} {pcy:.2f} c")
    parts.append(f"{pcx-r:.2f} {pcy-k:.2f} {pcx-k:.2f} {pcy-r:.2f} {pcx:.2f} {pcy-r:.2f} c")
    parts.append(f"{pcx+k:.2f} {pcy-r:.2f} {pcx+r:.2f} {pcy-k:.2f} {pcx+r:.2f} {pcy:.2f} c")
    parts.append("h")
    return (" ".join(parts) + "\n").encode()


def zone_diff_stats(gray: np.ndarray, zones: list[TintZone]) -> list[dict]:
    """Per-zone accuracy: false-negative (missed tint) and false-positive counts.

    Returns a list of dicts::
        zone_idx, rect, shape_type, tint_px,
        false_neg_px  (tint outside the polygon),
        false_pos_px  (non-tint inside the polygon),
        coverage_pct  (tint pixels covered / total tint pixels × 100),
        n_polygon_pts
    """
    h, w = gray.shape[:2]
    tint_global = (gray >= TINT_LO) & (gray <= TINT_HI)

    stats: list[dict] = []
    for i, zone in enumerate(zones):
        x0, y0, x1, y1 = zone.rect
        x0c = max(0, x0); y0c = max(0, y0)
        x1c = min(w, x1); y1c = min(h, y1)
        zh, zw = y1c - y0c, x1c - x0c
        if zh < 1 or zw < 1:
            continue

        tint_crop = tint_global[y0c:y1c, x0c:x1c]

        if zone.page_contour is not None:
            mask = np.zeros((zh, zw), dtype=np.uint8)
            cnt_crop = zone.page_contour.copy()
            cnt_crop[:, 0, 0] -= x0c
            cnt_crop[:, 0, 1] -= y0c
            cv2.fillPoly(mask, [cnt_crop], 255)
            in_shape = mask > 0
        else:
            in_shape = np.ones((zh, zw), dtype=bool)

        tint_px = int(tint_crop.sum())
        false_neg = int((tint_crop & ~in_shape).sum())
        false_pos = int((~tint_crop & in_shape).sum())
        cov = (tint_px - false_neg) / max(1, tint_px) * 100.0

        stats.append({
            "zone_idx": i,
            "rect": zone.rect,
            "shape_type": zone.shape_type,
            "tint_px": tint_px,
            "false_neg_px": false_neg,
            "false_pos_px": false_pos,
            "coverage_pct": round(cov, 1),
            "n_polygon_pts": len(zone.page_contour) if zone.page_contour is not None else 0,
        })
    return stats


# ---------------------------------------------------------------------------
# Shape classification helpers
# ---------------------------------------------------------------------------

def _estimate_corner_radius(tint_mask: np.ndarray, h: int, w: int) -> float:
    qh = max(1, h // 5)
    qw = max(1, w // 5)
    radii: list[float] = []
    corners = [
        (0, 0, qw, qh, "tl"),
        (w - qw, 0, w, qh, "tr"),
        (0, h - qh, qw, h, "bl"),
        (w - qw, h - qh, w, h, "br"),
    ]
    for cx0, cy0, cx1, cy1, cid in corners:
        region = tint_mask[cy0:cy1, cx0:cx1]
        missing_ys, missing_xs = np.where(region == 0)
        if missing_ys.size < 4:
            continue
        rh, rw = region.shape
        cy_frac = float(missing_ys.mean()) / max(1, rh - 1)
        cx_frac = float(missing_xs.mean()) / max(1, rw - 1)
        corner_cx, corner_cy = {
            "tl": (0.0, 0.0), "tr": (1.0, 0.0),
            "bl": (0.0, 1.0), "br": (1.0, 1.0),
        }[cid]
        dist_from_corner = (
            (cx_frac - corner_cx) ** 2 + (cy_frac - corner_cy) ** 2
        ) ** 0.5
        if dist_from_corner > 0.45:
            continue
        missing = int(missing_ys.size)
        r = float(np.sqrt(missing / (1.0 - np.pi / 4.0)))
        if r <= min(h, w) / 2:
            radii.append(r)
    return float(np.mean(radii)) if radii else 0.0
