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
  - Panels that are physically connected on the page (e.g. a header, a
    binding-shadow column and a footer that all touch at their corners,
    forming a frame around the body text) must be rendered as ONE shape with
    a hole, not as several separately-clipped rectangles stitched together --
    any artificial split leaves a visible seam or a double-blended overlap at
    the cut.

Architecture (layered compositing with Multiply blend):

  1. Bilevel base layer  -- Otsu-binarised.  Within the tint shape the
                           background is erased to white; text strokes are
                           identified with a zone-local threshold and kept black.
  2. Tint overlay        -- Greyscale JPEG image placed with /BM /Multiply,
                           clipped to the exact tint-shape contour via a PDF
                           clip path (W/W* n operator).
                           • tint background  → original grey value × 1.0 = tint
                           • paper (> TINT_HI) → 255 × 1.0 = white (identity)
                           • text stroke area → any × 0 = 0 (stays black)

Whole-page, hierarchy-aware contour detection:
  1. Compute tint_mask = pixels in [TINT_LO, TINT_HI] on the full grey image.
  2. Apply a modest morphological CLOSE (a small, fixed pixel radius -- not
     scaled to page size) to bridge text-stroke holes within a tint panel
     without eroding real shape features like rounded corners.
  3. cv2.findContours(RETR_CCOMP) on the WHOLE PAGE in one pass gives a
     2-level hierarchy: each top-level (outer) contour plus any holes
     directly inside it.  If a header, a column and a footer are physically
     touching, MORPH_CLOSE merges them into ONE connected component whose
     outer contour is the frame's outline and whose hole is the body-text
     cavity it encloses -- exactly the topology of the source artwork, with
     no arbitrary row/column split and therefore no seam.
  4. Holes much smaller than a plausible body-text cavity (i.e. ordinary
     text-stroke gaps that MORPH_CLOSE didn't fully bridge) are discarded so
     they don't get carved out of the tint as if they were real cavities.
  5. cv2.approxPolyDP simplifies each outer/hole contour to a manageable
     polyline (epsilon capped in pixels, not scaled to perimeter, so a small
     curved feature on a very large combined contour still gets enough
     vertices).  A small dilate/erode pass beforehand removes the hairline
     notch that can otherwise appear where a curve is tangent to a flat edge.
  6. The outer polyline is emitted as PDF "m/l/h" path operators; each hole's
     polyline is appended as an additional subpath in the SAME path object,
     and the clip operator is "W*" (even-odd) instead of "W" (nonzero) so the
     hole is excluded from the painted area -- the standard PDF technique for
     clipping to a shape with a cutout ("donut" clipping).

Why not just keep the previous per-panel approach?  ``detect_tint_panels``
hands us independent rectangular hints (header / footer / column) and the
earlier implementation processed each hint in its own local crop to avoid a
historical bug: closing the WHOLE page with a large kernel merged the three
into one connected blob, and cv2.RETR_EXTERNAL discards a contour's holes --
so cv2.fillPoly on that single outer contour filled the body-text cavity
solid, not just the thin frame.  The bug was never "merging is wrong"; it was
that the hole information was being thrown away.  Once RETR_CCOMP keeps that
hole, the panels can be processed together -- which is also strictly simpler
than the previous per-orientation expansion / seam-margin machinery.
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

# Whole-page MORPH_CLOSE kernel radius, in pixels -- a FIXED absolute value,
# not scaled to page size.  Its job is to bridge ordinary text-stroke holes
# (a property of stroke width / scan resolution, not of how big the page is);
# scaling it to page dimensions made it large enough to erode real rounded
# corners on a typical-sized scan.
_CLOSE_PX: int = 12
_CLOSE_ITERATIONS: int = 2

# Contour simplification: fraction of arc length used as epsilon for
# cv2.approxPolyDP, capped at _MAX_APPROX_ERR_PX regardless of perimeter so a
# small curved feature (e.g. a circular badge) on a very large combined
# contour (e.g. header+column+footer merged into one frame) still keeps
# enough vertices to look smooth.
_APPROX_FRAC: float = 0.003
_MAX_APPROX_ERR_PX: float = 3.0

# Pad contours by this many pixels (dilate outer / erode holes) before
# simplifying.  At a tangent point where a curve meets a flat edge,
# approxPolyDP's vertex placement can leave a hairline notch a few px wide;
# padding removes it at the cost of a few extra paper pixels marked in-shape
# (harmless: Multiply renders paper as identity).
_PAD_PX: int = 2

# Circle detection: if a (hole-free) contour's compactness exceeds this
# threshold it's treated as a circle and clipped with exact Bézier arcs
# rather than a polygon.
_CIRCULARITY_MIN: float = 0.85

# A hole must occupy at least this fraction of the page area (with an
# absolute pixel floor) to be treated as a real structural cavity -- e.g. the
# body-text area enclosed by a header/footer/column frame -- rather than an
# ordinary text-stroke gap that MORPH_CLOSE left unbridged.
_MIN_HOLE_AREA_FRAC: float = 0.0005
_MIN_HOLE_AREA_PX: int = 2000

# An outer contour is only promoted to a TintZone if it overlaps at least one
# detect_tint_panels hint by this fraction of the hint's own area -- filters
# out unrelated tint-coloured noise elsewhere on the page.
_MIN_PANEL_OVERLAP_FRAC: float = 0.3


@dataclass
class TintZone:
    """Rendering data for one tint shape (possibly with one or more holes)."""

    rect: tuple[int, int, int, int]      # (x0, y0, x1, y1) in page-image pixels
    overlay_img: np.ndarray              # HxW uint8 greyscale; 255 = transparent
    page_contour: np.ndarray | None = None  # (N,1,2) int32, page-image space (y-down)
    hole_contours: list = field(default_factory=list)  # list of (N,1,2) int32 holes
    shape_type: str = "rect"             # "rect" | "rounded_rect" | "poly" | "circle" | "frame"
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

    ``panels`` is the output of ``tint_panel.detect_tint_panels`` and is used
    only as a hint to decide which connected tint components on the page are
    worth turning into zones (filters out unrelated tint-coloured noise).

    Strategy: one whole-page MORPH_CLOSE + RETR_CCOMP pass finds every
    connected tint component together with its holes (see module docstring).
    Physically touching panels (header/column/footer forming a frame around
    the body text) therefore come back as ONE zone with one hole -- no
    artificial seam between them.  A panel hint with no matching component
    (e.g. too small / isolated for the synthetic-test cases) falls back to a
    plain rectangular zone.

    Returns a list of TintZone objects.
    """
    if not panels:
        return []

    h, w = gray.shape[:2]
    page_area = h * w

    tint_mask = ((gray >= TINT_LO) & (gray <= TINT_HI)).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_CLOSE_PX * 2 + 1, _CLOSE_PX * 2 + 1)
    )
    closed = cv2.morphologyEx(
        tint_mask, cv2.MORPH_CLOSE, kernel, iterations=_CLOSE_ITERATIONS
    )

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    zones: list[TintZone] = []
    claimed = [False] * len(panels)

    if contours:
        hier = hierarchy[0]
        min_hole_area = max(_MIN_HOLE_AREA_PX, int(page_area * _MIN_HOLE_AREA_FRAC))

        for i, info in enumerate(hier):
            parent = info[3]
            if parent != -1:
                continue  # a hole; collected below together with its outer parent
            outer_cnt = contours[i]
            if cv2.contourArea(outer_cnt) < 16:
                continue

            overlap_idx = _overlapping_panel_indices(outer_cnt, panels)
            if not overlap_idx:
                continue
            for idx in overlap_idx:
                claimed[idx] = True

            holes: list[np.ndarray] = []
            child = info[2]
            while child != -1:
                hole_cnt = contours[child]
                if cv2.contourArea(hole_cnt) >= min_hole_area:
                    holes.append(hole_cnt)
                child = hier[child][0]

            zone = _make_zone_from_components(gray, outer_cnt, holes, w, h)
            if zone is not None:
                zones.append(zone)

    # Fallback: any hint with no matching connected component (too small or
    # isolated for MORPH_CLOSE/area thresholds to pick up) still gets a
    # simple rectangular zone so it isn't silently dropped.
    for idx, p in enumerate(panels):
        if claimed[idx]:
            continue
        z = _make_tint_zone_rect(gray, p["rect"])
        if z is not None:
            zones.append(z)

    return zones


def make_tint_overlay(gray_crop: np.ndarray) -> np.ndarray:
    """Return the overlay image for a tint panel crop.

    Paper pixels (> TINT_HI) are set to 255 so Multiply has no effect there.
    All other pixels keep their original grey value.
    """
    overlay = gray_crop.copy()
    overlay[gray_crop > TINT_HI] = 255
    return overlay


def contour_to_pdf_path(
    contour: np.ndarray,
    page_h_px: int,
    holes: list | None = None,
) -> bytes:
    """Convert an outer contour (plus optional holes) to PDF path bytes.

    Coordinate conversion: image space (y-down) → PDF space (y-up).
      pdf_y = page_h_px - image_y

    Returns bytes suitable for insertion before a clip operator in a PDF
    content stream.  Each contour becomes its own "m / l ... / h" subpath;
    when holes are present the caller MUST use the even-odd clip operator
    ("W* n") rather than nonzero ("W n") so the hole subpaths are excluded
    from the clipped (painted) region -- the standard PDF "donut clip"
    technique. All coordinates are integer pixel values; the caller's
    responsibility to apply any further scale transform (e.g. the 72/dpi
    scale set by _set_physical_page_size wraps the whole page in a cm, so
    nothing extra is needed here).
    """
    def _emit(cnt: np.ndarray) -> str | None:
        if cnt is None or len(cnt) < 3:
            return None
        parts: list[str] = []
        for i, pt in enumerate(cnt):
            x = int(pt[0][0])
            y = page_h_px - int(pt[0][1])   # y-up
            parts.append(f"{x} {y} m" if i == 0 else f"{x} {y} l")
        parts.append("h")
        return " ".join(parts)

    outer = _emit(contour)
    if outer is None:
        return b""
    pieces = [outer]
    for hole in (holes or []):
        piece = _emit(hole)
        if piece is not None:
            pieces.append(piece)
    return ("\n".join(pieces) + "\n").encode()


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

def _overlapping_panel_indices(cnt: np.ndarray, panels: list[dict]) -> list[int]:
    """Indices of panel hints that this contour overlaps by enough to count.

    The overlap fraction is taken against the SMALLER of the panel hint's
    own area and the contour's bounding-box area (not just the hint's area).
    A small badge attached to (or near) a large header hint can overlap that
    header by a tiny fraction of the header's own huge area while still
    being entirely inside it -- checking only against the hint's area would
    incorrectly reject it as "unrelated noise".
    """
    bx, by, bw, bh = cv2.boundingRect(cnt)
    cnt_area = max(1, bw * bh)
    found: list[int] = []
    for idx, p in enumerate(panels):
        px0, py0, px1, py1 = p["rect"]
        ix0, iy0 = max(bx, px0), max(by, py0)
        ix1, iy1 = min(bx + bw, px1), min(by + bh, py1)
        if ix1 > ix0 and iy1 > iy0:
            inter = (ix1 - ix0) * (iy1 - iy0)
            panel_area = max(1, (px1 - px0) * (py1 - py0))
            denom = min(panel_area, cnt_area)
            if inter / denom >= _MIN_PANEL_OVERLAP_FRAC:
                found.append(idx)
    return found


def _pad_and_simplify(
    cnt: np.ndarray,
    page_h: int,
    page_w: int,
    grow: bool,
) -> np.ndarray:
    """Dilate (grow=True, for outer contours) or erode (grow=False, for
    holes) by _PAD_PX before approxPolyDP, removing hairline notches at
    tangent points between curves and flat edges; then simplify."""
    bx, by, bw, bh = cv2.boundingRect(cnt)
    pad = _PAD_PX + 2
    x0 = max(0, bx - pad); y0 = max(0, by - pad)
    x1 = min(page_w, bx + bw + pad); y1 = min(page_h, by + bh + pad)
    local = cnt.copy()
    local[:, 0, 0] -= x0
    local[:, 0, 1] -= y0

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.drawContours(mask, [local], -1, 255, thickness=cv2.FILLED)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_PAD_PX * 2 + 1, _PAD_PX * 2 + 1))
    mask = cv2.dilate(mask, k, iterations=1) if grow else cv2.erode(mask, k, iterations=1)

    padded, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = max(padded, key=cv2.contourArea) if padded else local

    peri = cv2.arcLength(best, True)
    epsilon = max(1.5, min(_MAX_APPROX_ERR_PX, peri * _APPROX_FRAC))
    approx = cv2.approxPolyDP(best, epsilon, True)
    approx[:, 0, 0] += x0
    approx[:, 0, 1] += y0
    return approx


def _make_zone_from_components(
    gray: np.ndarray,
    outer_cnt: np.ndarray,
    holes: list[np.ndarray],
    page_w: int,
    page_h: int,
) -> TintZone | None:
    outer = _pad_and_simplify(outer_cnt, page_h, page_w, grow=True)
    hole_polys = [_pad_and_simplify(hc, page_h, page_w, grow=False) for hc in holes]

    bx, by, bw, bh = cv2.boundingRect(outer)
    x0 = max(0, bx); y0 = max(0, by)
    x1 = min(page_w, bx + bw); y1 = min(page_h, by + bh)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    overlay = make_tint_overlay(gray[y0:y1, x0:x1])

    circle_fit: tuple[float, float, float] | None = None
    if not hole_polys and _contour_circularity(outer_cnt) >= _CIRCULARITY_MIN:
        (cx, cy), r = cv2.minEnclosingCircle(outer_cnt)
        circle_fit = (float(cx), float(cy), float(r))

    if circle_fit is not None:
        shape_type = "circle"
        corner_radius = 0.0
    elif hole_polys:
        shape_type = "frame"
        corner_radius = 0.0
    else:
        gray_crop = gray[y0:y1, x0:x1]
        tint_mask = ((gray_crop >= TINT_LO) & (gray_crop <= TINT_HI)).astype(np.uint8)
        shape_type, corner_radius, _ = classify_tint_shape(tint_mask)
        if len(outer) > 8:
            shape_type = "poly"

    return TintZone(
        rect=(x0, y0, x1, y1),
        overlay_img=overlay,
        page_contour=outer,
        hole_contours=hole_polys,
        shape_type=shape_type,
        corner_radius=corner_radius,
        contour_pts=[],
        circle_fit=circle_fit,
    )


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

    Holes are subtracted from the in-shape mask, matching the actual rendered
    (even-odd clipped / hole-punched) coverage.

    Returns a list of dicts::
        zone_idx, rect, shape_type, tint_px,
        false_neg_px  (tint outside the polygon, or inside a hole),
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
            for hole in zone.hole_contours:
                hole_crop = hole.copy()
                hole_crop[:, 0, 0] -= x0c
                hole_crop[:, 0, 1] -= y0c
                cv2.fillPoly(mask, [hole_crop], 0)
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
