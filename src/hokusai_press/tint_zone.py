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

# Vector tint fills are only worthwhile for very smooth, low-complexity tone
# panels.  Real scanned halftone noise can quantize into thousands of tiny
# islands; keep these limits deliberately conservative so JPEG remains the
# fallback whenever polygon output is likely to grow the PDF.
_VEC_TINT_SMOOTH_RADIUS_PX: int = 3
_VEC_TINT_MIN_COMPONENT_AREA_PX: int = 64
_VEC_TINT_MIN_COMPONENT_AREA_FRAC: float = 0.0001
_VEC_TINT_MAX_COMPONENT_AREA_PX: int = 1000
_VEC_TINT_MAX_COMPONENTS: int = 50
_VEC_TINT_MAX_VERTICES: int = 500
_VEC_TINT_KMEANS_SAMPLE_MAX: int = 200_000
_VEC_TINT_MAX_DRAWABLE_PIXELS: int = 750_000

# Large scanned halftone areas are often semantically one flat printed colour:
# the dot pattern is a printing/optical artifact, not artwork that must be
# preserved as thousands of tiny polygons.  This flat-tone path deliberately
# looks at a low-frequency version of the zone, then emits only large regions.
_VEC_FLAT_MIN_DRAWABLE_PIXELS: int = 100_000
_VEC_FLAT_DOWNSCALE_MAX_DIM: int = 900
_VEC_FLAT_MAX_K: int = 4
_VEC_FLAT_QUANT_GOOD_MIN: float = 0.70
_VEC_FLAT_TILE_PX: int = 64
_VEC_FLAT_TILE_MEDIAN_IQR_MAX: float = 28.0
_VEC_FLAT_MIN_COMPONENT_AREA_PX: int = 4_000
_VEC_FLAT_MIN_COMPONENT_AREA_FRAC: float = 0.0002
_VEC_FLAT_MAX_COMPONENTS: int = 28
_VEC_FLAT_MAX_VERTICES: int = 6_000
_VEC_FLAT_CLOSE_PX: int = 4
_VEC_FLAT_BRIGHT_HOLE_MIN_AREA_PX: int = 24
_VEC_PATCH_PAD_PX: int = 6
_VEC_PATCH_MAX_RECTS: int = 80

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

# Small holes are usually text-stroke gaps, but OCR text boxes can identify a
# deliberate paper cavity inside a tint frame even when the cavity is below the
# global area threshold (common after crop/scale normalization).
_HOLE_TEXT_OVERLAP_FRAC: float = 0.15

# A hole must also occupy at least this fraction of its OWN outer shape's
# area, regardless of the page-relative checks above.  Without this, a
# decorative caption inside a small zone (e.g. a badge) can exceed the
# page-relative floor purely because the page is large, even though the
# caption is a small sliver of the badge itself -- see build_tint_zones.
#
# A genuine cavity enclosed by a thin frame (e.g. a header/column/footer band
# around the body text) is typically much LARGER in area than the thin band
# enclosing it -- a real example measured well over 100% of its own outer
# shape's area.  A badge caption's merged-by-closing text blob measured only
# ~15-25%.  0.5 sits well inside that gap.
_MIN_HOLE_FRAC_OF_ZONE: float = 0.5

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
    vectorize_success: bool = False
    vectorized_fills: list[dict] = field(default_factory=list)
    vectorized_patches: list[tuple[int, int, int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tint_zones(
    gray: np.ndarray,
    panels: list[dict],
    text_boxes: list[tuple[float, float, float, float]] | None = None,
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
    if text_boxes:
        # Dense body text's own anti-aliased pixels routinely land in
        # [TINT_LO, TINT_HI]; over a packed paragraph that can chain-bridge
        # through MORPH_CLOSE across enough distance to fuse a real tint
        # panel with an unrelated chunk of body text far away (found by
        # comparing actual render output to the source scan -- the fused
        # area then gets a Multiply overlay it was never meant to have).
        # Zeroing OCR text-box pixels before closing removes them from the
        # connectivity analysis entirely: a text box small enough to sit
        # fully inside a real tint panel (e.g. a heading printed on a
        # header band) still gets bridged over like any other text-stroke
        # gap, but a body paragraph far from any panel no longer offers a
        # path to bridge through.
        text_mask = _boxes_to_mask(text_boxes, h, w)
        tint_mask[text_mask] = 0
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

            outer_area = max(1.0, cv2.contourArea(outer_cnt))
            holes: list[np.ndarray] = []
            child = info[2]
            while child != -1:
                hole_cnt = contours[child]
                hole_area = cv2.contourArea(hole_cnt)
                # A real cavity (e.g. the body-text area enclosed by a
                # header/column/footer frame) takes up a substantial share of
                # its OWN outer shape's area.  A small decorative zone (e.g.
                # a badge) can have a caption whose merged-by-closing white
                # knockout blob exceeds the page-relative area floor purely
                # because the page is large, while still being a small sliver
                # of that badge's own area -- punching it out as a "hole"
                # would skip ink/tint classification for any tint-coloured
                # gaps the closing swept into that blob, exposing raw
                # (un-tinted) Otsu bilevel underneath.  Requiring a minimum
                # share of the outer shape's own area rejects those without
                # affecting genuine cavities, which comfortably clear it.
                large_enough_share = hole_area / outer_area >= _MIN_HOLE_FRAC_OF_ZONE
                if large_enough_share and (
                    hole_area >= min_hole_area
                    or _hole_overlaps_text_box(hole_cnt, text_boxes)
                ):
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

def _boxes_to_mask(
    boxes: list[tuple[float, float, float, float]],
    h: int,
    w: int,
) -> np.ndarray:
    """Rasterize a list of (x0,y0,x1,y1) boxes into a boolean (h, w) mask."""
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        if len(box) < 4:
            continue
        x0f, y0f, x1f, y1f = (float(v) for v in box[:4])
        x0 = max(0, min(w, int(np.floor(min(x0f, x1f)))))
        y0 = max(0, min(h, int(np.floor(min(y0f, y1f)))))
        x1 = max(0, min(w, int(np.ceil(max(x0f, x1f)))))
        y1 = max(0, min(h, int(np.ceil(max(y0f, y1f)))))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


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


def _hole_overlaps_text_box(
    hole_cnt: np.ndarray,
    text_boxes: list[tuple[float, float, float, float]] | None,
) -> bool:
    """True when a small contour hole aligns with OCR text-region geometry.

    The comparison intentionally uses bounding boxes instead of contour masks:
    OCR regions are rectangular hints already in rendered-pixel coordinates,
    and this mirrors the cheap intersection pattern used for panel hints above.
    """
    if not text_boxes:
        return False

    bx, by, bw, bh = cv2.boundingRect(hole_cnt)
    hole_area = max(1.0, float(bw * bh))
    for box in text_boxes:
        if len(box) < 4:
            continue
        tx0, ty0, tx1, ty1 = (float(v) for v in box[:4])
        x0, x1 = sorted((tx0, tx1))
        y0, y1 = sorted((ty0, ty1))
        text_area = max(1.0, (x1 - x0) * (y1 - y0))
        ix0, iy0 = max(float(bx), x0), max(float(by), y0)
        ix1, iy1 = min(float(bx + bw), x1), min(float(by + bh), y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        inter = (ix1 - ix0) * (iy1 - iy0)
        if inter / min(hole_area, text_area) >= _HOLE_TEXT_OVERLAP_FRAC:
            return True
    return False


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


def tint_render_quality(
    orig_gray: np.ndarray,
    rendered_gray: np.ndarray,
    text_boxes: list[tuple] | None = None,
    tile: int = 32,
    deviation_threshold: float = 40.0,
    zone_mask: np.ndarray | None = None,
) -> list[dict]:
    """Find local tint-rendering failures while ignoring OCR text regions.

    ``zone_diff_stats`` is polygon-coverage oriented and can count antialiased
    text pixels as missed tint.  This diagnostic instead compares mean grey on
    tint-heavy tiles after masking OCR text boxes, so intentionally bilevel text
    does not look like a tint rendering error.

    ``zone_mask``, if given, is a boolean (h, w) array marking pixels that are
    actually inside a rendered TintZone's painted area (e.g. the union of each
    zone's polygon minus its holes).  Real OCR coverage is often incomplete --
    a page can have dense body-text tiles where >= 50% of pixels happen to
    land in [TINT_LO, TINT_HI] purely from anti-aliasing, with no matching
    text_box to exclude them.  Restricting evaluation to ``zone_mask`` (where
    given) ties this diagnostic to "did we render the area we intended to
    treat as tint", which is robust to OCR gaps; without it, the looser
    >=50%-tint-pixel heuristic is used as a fallback.
    """
    if orig_gray.shape != rendered_gray.shape:
        raise ValueError("orig_gray and rendered_gray must have the same shape")
    if orig_gray.ndim != 2 or rendered_gray.ndim != 2:
        raise ValueError("orig_gray and rendered_gray must be 2-D grayscale images")
    if tile < 1:
        raise ValueError("tile must be >= 1")
    if zone_mask is not None and zone_mask.shape != orig_gray.shape:
        raise ValueError("zone_mask must have the same shape as orig_gray")

    h, w = orig_gray.shape
    eval_mask = np.ones((h, w), dtype=bool) if zone_mask is None else zone_mask.astype(bool).copy()
    if text_boxes:
        eval_mask &= ~_boxes_to_mask(text_boxes, h, w)

    issues: list[dict] = []
    for y0 in range(0, h, tile):
        y1 = min(h, y0 + tile)
        for x0 in range(0, w, tile):
            x1 = min(w, x0 + tile)
            valid = eval_mask[y0:y1, x0:x1]
            valid_px = int(valid.sum())
            if valid_px == 0:
                continue

            orig_tile = orig_gray[y0:y1, x0:x1]
            rendered_tile = rendered_gray[y0:y1, x0:x1]
            if zone_mask is None:
                tint_px = ((orig_tile >= TINT_LO) & (orig_tile <= TINT_HI) & valid)
                if float(tint_px.sum()) / valid_px < 0.5:
                    continue

            orig_mean = float(orig_tile[valid].mean())
            rendered_mean = float(rendered_tile[valid].mean())
            deviation = rendered_mean - orig_mean
            if abs(deviation) < deviation_threshold:
                continue

            issues.append({
                "rect": (x0, y0, x1, y1),
                "orig_mean": orig_mean,
                "rendered_mean": rendered_mean,
                "deviation": deviation,
                "kind": "too_light" if deviation > 0 else "too_dark",
            })

    issues.sort(key=lambda item: abs(item["deviation"]), reverse=True)
    return issues


def zone_mask_from_zones(zones: list[TintZone], page_h: int, page_w: int) -> np.ndarray:
    """Union of every zone's painted area (polygon minus its holes).

    Convenience helper for ``tint_render_quality(..., zone_mask=...)`` so
    callers don't have to re-derive the fillPoly/hole-punch logic that
    ``mrc.py`` and ``zone_diff_stats`` already use.
    """
    mask = np.zeros((page_h, page_w), dtype=np.uint8)
    for zone in zones:
        x0, y0, x1, y1 = zone.rect
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(page_w, x1), min(page_h, y1)
        if x1c - x0c < 1 or y1c - y0c < 1:
            continue
        sub = mask[y0c:y1c, x0c:x1c]
        if zone.page_contour is not None:
            cnt = zone.page_contour.copy()
            cnt[:, 0, 0] -= x0c
            cnt[:, 0, 1] -= y0c
            cv2.fillPoly(sub, [cnt], 255)
            for hole in zone.hole_contours:
                hc = hole.copy()
                hc[:, 0, 0] -= x0c
                hc[:, 0, 1] -= y0c
                cv2.fillPoly(sub, [hc], 0)
        else:
            sub[:] = 255
    return mask > 0


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


def try_vectorize_zone(
    zone: TintZone,
    crop_gray: np.ndarray,
    in_shape: np.ndarray,
    ink: np.ndarray,
    protect_mask: np.ndarray | None = None,
) -> bool:
    """平網タイントをPDFベクター図形に量子化・ポリゴン化可能か検証し、可能ならベクター化データを設定する。"""
    overlay_img = zone.overlay_img
    zone.vectorize_success = False
    zone.vectorized_fills = []
    zone.vectorized_patches = []

    protected = (
        np.zeros_like(in_shape, dtype=bool)
        if protect_mask is None
        else (protect_mask & in_shape)
    )
    valid_mask = in_shape & ~ink & ~protected
    pixels = overlay_img[valid_mask]

    # 255 (紙の白) は Multiplyブレンドで描画不要なので、ベクター化の量子化対象からは除外する
    pixels = pixels[pixels < 255]

    if pixels.size < 100:
        return False
    if pixels.size >= _VEC_FLAT_MIN_DRAWABLE_PIXELS:
        if _try_vectorize_flat_tone_zone(
            zone, overlay_img, valid_mask, pixels, protected
        ):
            return True
    if pixels.size > _VEC_TINT_MAX_DRAWABLE_PIXELS:
        return False

    best_K = None
    best_centers = None
    best_reconstructed = None

    if pixels.size > _VEC_TINT_KMEANS_SAMPLE_MAX:
        step = int(np.ceil(pixels.size / _VEC_TINT_KMEANS_SAMPLE_MAX))
        kmeans_pixels = pixels[::step]
    else:
        kmeans_pixels = pixels
    data = kmeans_pixels.astype(np.float32).reshape(-1, 1)
    drawable_total = max(1, int(pixels.size))

    for K in range(2, 7):
        if kmeans_pixels.size < K:
            break
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_PP_CENTERS
        try:
            compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flags)
        except Exception:
            continue

        reconstructed = np.full_like(overlay_img, 255)
        drawable = valid_mask & (overlay_img < 255)
        full_pixels = overlay_img[drawable].astype(np.float32)
        center_vals = centers.flatten().astype(np.float32)
        best_dist = np.full(full_pixels.shape, np.inf, dtype=np.float32)
        quantized_pixels = np.empty(full_pixels.shape, dtype=np.float32)
        for c in center_vals:
            dist = np.abs(full_pixels - c)
            take = dist < best_dist
            quantized_pixels[take] = c
            best_dist[take] = dist[take]
        reconstructed[drawable] = np.clip(
            np.round(quantized_pixels), 0, 255
        ).astype(np.uint8)

        diff = np.abs(overlay_img.astype(np.int32) - reconstructed.astype(np.int32))
        good_pixels = int(np.sum(diff[drawable] <= 15))

        if (good_pixels / drawable_total) >= 0.95:
            best_K = K
            best_centers = centers
            best_reconstructed = reconstructed
            break

    if best_K is None:
        return False

    zone_h, zone_w = overlay_img.shape[:2]
    zone_area = max(1, zone_h * zone_w)
    min_component_area = max(
        _VEC_TINT_MIN_COMPONENT_AREA_PX,
        min(
            _VEC_TINT_MAX_COMPONENT_AREA_PX,
            int(round(zone_area * _VEC_TINT_MIN_COMPONENT_AREA_FRAC)),
        ),
    )
    smooth_k = _VEC_TINT_SMOOTH_RADIUS_PX * 2 + 1
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_k, smooth_k))
    unique_centers = sorted(best_centers.flatten())

    cleaned_masks: list[tuple[int, np.ndarray]] = []
    for c in unique_centers:
        c_int = int(round(c))
        if c_int >= 255:
            continue

        color_mask = (best_reconstructed == c_int).astype(np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, smooth_kernel, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, smooth_kernel, iterations=1)
        color_mask = (color_mask > 0) & valid_mask
        if int(color_mask.sum()) < min_component_area:
            continue
        cleaned_masks.append((c_int, color_mask))

    if not cleaned_masks:
        return False

    drawable_mask = valid_mask & (overlay_img < 255)
    assigned = np.zeros((zone_h, zone_w), dtype=bool)
    smoothed = np.full_like(overlay_img, 255)
    for c_int, color_mask in sorted(cleaned_masks, key=lambda item: int(item[1].sum()), reverse=True):
        take = color_mask & ~assigned
        smoothed[take] = c_int
        assigned[take] = True

    missing = drawable_mask & ~assigned
    if missing.any():
        dist_maps = []
        for _, color_mask in cleaned_masks:
            if color_mask.any():
                dist_maps.append(cv2.distanceTransform((~color_mask).astype(np.uint8), cv2.DIST_L2, 3))
            else:
                dist_maps.append(np.full((zone_h, zone_w), np.inf, dtype=np.float32))
        nearest = np.argmin(np.stack(dist_maps, axis=0), axis=0)
        for i, (c_int, _) in enumerate(cleaned_masks):
            smoothed[missing & (nearest == i)] = c_int

    smooth_diff = np.abs(overlay_img.astype(np.int32) - smoothed.astype(np.int32))
    smooth_good = int(np.sum(smooth_diff[drawable_mask] <= 20))
    if smooth_good / max(1, int(drawable_mask.sum())) < 0.95:
        return False

    vector_fills = []
    total_components = 0
    total_vertices = 0

    for c_int, _ in cleaned_masks:
        color_mask = ((smoothed == c_int) & drawable_mask).astype(np.uint8)
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        hier = hierarchy[0]
        for idx, info in enumerate(hier):
            parent = info[3]
            if parent != -1:
                continue
            outer_cnt = contours[idx]
            if cv2.contourArea(outer_cnt) < min_component_area:
                continue

            holes = []
            child = info[2]
            while child != -1:
                hole_cnt = contours[child]
                if cv2.contourArea(hole_cnt) >= min_component_area:
                    holes.append(hole_cnt)
                child = hier[child][0]

            try:
                simplified_outer = _pad_and_simplify(outer_cnt, zone_h, zone_w, grow=True)
                simplified_holes = [
                    _pad_and_simplify(hc, zone_h, zone_w, grow=False)
                    for hc in holes
                ]
            except Exception:
                simplified_outer = outer_cnt
                simplified_holes = holes

            total_components += 1
            total_vertices += len(simplified_outer) + sum(len(hc) for hc in simplified_holes)
            if (
                total_components > _VEC_TINT_MAX_COMPONENTS
                or total_vertices > _VEC_TINT_MAX_VERTICES
            ):
                return False

            x0, y0 = zone.rect[:2]

            page_outer = simplified_outer.copy()
            page_outer[:, 0, 0] += x0
            page_outer[:, 0, 1] += y0

            page_holes = []
            for hc in simplified_holes:
                page_hc = hc.copy()
                page_hc[:, 0, 0] += x0
                page_hc[:, 0, 1] += y0
                page_holes.append(page_hc)

            gray_val = float(c_int) / 255.0

            vector_fills.append({
                "gray": gray_val,
                "contour": page_outer,
                "holes": page_holes,
            })

    if not vector_fills:
        return False

    zone.vectorize_success = True
    zone.vectorized_fills = vector_fills
    return True


def _try_vectorize_flat_tone_zone(
    zone: TintZone,
    overlay_img: np.ndarray,
    valid_mask: np.ndarray,
    pixels: np.ndarray,
    protected: np.ndarray,
) -> bool:
    """Vectorize large halftone regions as a few flat printed tone areas."""
    h, w = overlay_img.shape[:2]
    drawable = valid_mask & (overlay_img < 255)
    if int(drawable.sum()) < _VEC_FLAT_MIN_DRAWABLE_PIXELS:
        return False

    max_dim = max(h, w)
    scale = min(1.0, _VEC_FLAT_DOWNSCALE_MAX_DIM / float(max_dim))
    if scale < 1.0:
        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        small_img = cv2.resize(
            overlay_img, (small_w, small_h), interpolation=cv2.INTER_AREA
        )
        small_drawable = cv2.resize(
            drawable.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST
        ) > 0
    else:
        small_img = overlay_img
        small_drawable = drawable

    small_pixels = small_img[small_drawable]
    small_pixels = small_pixels[small_pixels < 255]
    if small_pixels.size < 100:
        return False

    sample = (
        small_pixels[:: int(np.ceil(small_pixels.size / _VEC_TINT_KMEANS_SAMPLE_MAX))]
        if small_pixels.size > _VEC_TINT_KMEANS_SAMPLE_MAX
        else small_pixels
    )
    data = sample.astype(np.float32).reshape(-1, 1)

    best: tuple[np.ndarray, np.ndarray, float] | None = None
    full_small = small_img[small_drawable].astype(np.float32)
    for k in range(1, _VEC_FLAT_MAX_K + 1):
        if sample.size < k:
            break
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, _, centers = cv2.kmeans(
                data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
            )
        except Exception:
            continue
        center_vals = centers.flatten().astype(np.float32)
        labels = _nearest_gray_labels(full_small, center_vals)
        quant = center_vals[labels]
        good = float((np.abs(full_small - quant) <= 25).mean())
        if good >= _VEC_FLAT_QUANT_GOOD_MIN:
            best = (center_vals, labels, good)
    if best is None:
        return False

    center_vals, labels_1d, good = best
    if good < _VEC_FLAT_QUANT_GOOD_MIN:
        return False

    label_img = np.full(small_img.shape, -1, dtype=np.int16)
    label_img[small_drawable] = labels_1d.astype(np.int16)

    min_area = max(
        _VEC_FLAT_MIN_COMPONENT_AREA_PX,
        int(round(h * w * _VEC_FLAT_MIN_COMPONENT_AREA_FRAC)),
    )
    small_min_area = max(4, int(round(min_area * scale * scale)))
    close_k = max(3, _VEC_FLAT_CLOSE_PX * 2 + 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    vector_fills: list[dict] = []
    total_components = 0
    total_vertices = 0
    bright_holes = _bright_knockout_hole_contours(overlay_img, valid_mask)
    protected_holes = _protected_hole_contours(protected)

    order = sorted(range(len(center_vals)), key=lambda i: float(center_vals[i]))
    for label in order:
        small_mask = ((label_img == label) & small_drawable).astype(np.uint8)
        if int(small_mask.sum()) < small_min_area:
            continue
        small_mask = cv2.morphologyEx(
            small_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1
        )
        small_mask = cv2.morphologyEx(
            small_mask, cv2.MORPH_OPEN, close_kernel, iterations=1
        )
        small_mask = (small_mask > 0) & small_drawable
        if int(small_mask.sum()) < small_min_area:
            continue
        mask = cv2.resize(
            small_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ) > 0
        mask &= drawable
        if int(mask.sum()) < min_area:
            continue
        if not _flat_component_distribution_ok(overlay_img, mask):
            continue

        contours, hierarchy = cv2.findContours(
            small_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        if hierarchy is None:
            continue
        hier = hierarchy[0]
        for idx, info in enumerate(hier):
            if info[3] != -1:
                continue
            outer_cnt = contours[idx]
            if cv2.contourArea(outer_cnt) < small_min_area:
                continue
            holes = []
            child = info[2]
            while child != -1:
                hole_cnt = contours[child]
                if cv2.contourArea(hole_cnt) >= small_min_area:
                    holes.append(hole_cnt)
                child = hier[child][0]

            simplified_outer = _scale_simplified_contour(outer_cnt, scale, w, h)
            simplified_holes = [
                _scale_simplified_contour(hc, scale, w, h) for hc in holes
            ]
            simplified_holes.extend(_holes_inside_mask(bright_holes, mask))
            simplified_holes.extend(_holes_inside_mask(protected_holes, mask))

            total_components += 1
            total_vertices += len(simplified_outer) + sum(len(hc) for hc in simplified_holes)
            if (
                total_components > _VEC_FLAT_MAX_COMPONENTS
                or total_vertices > _VEC_FLAT_MAX_VERTICES
            ):
                return False

            x0, y0 = zone.rect[:2]
            page_outer = simplified_outer.copy()
            page_outer[:, 0, 0] += x0
            page_outer[:, 0, 1] += y0
            page_holes = []
            for hc in simplified_holes:
                page_hc = hc.copy()
                page_hc[:, 0, 0] += x0
                page_hc[:, 0, 1] += y0
                page_holes.append(page_hc)

            component_pixels = overlay_img[(mask > 0) & (overlay_img < 255)]
            gray_val = (
                float(np.median(component_pixels)) / 255.0
                if component_pixels.size
                else float(center_vals[label]) / 255.0
            )
            vector_fills.append({
                "gray": gray_val,
                "contour": page_outer,
                "holes": page_holes,
            })

    if not vector_fills:
        return False
    zone.vectorize_success = True
    zone.vectorized_fills = vector_fills
    zone.vectorized_patches = []
    return True


def _protected_hole_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Convert OCR-protected knockout text pixels into vector fill holes."""
    if not mask.any():
        return []
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    holes: list[np.ndarray] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 4:
            continue
        simplified = cv2.approxPolyDP(cnt, 1.2, True)
        if len(simplified) >= 3:
            holes.append(simplified.astype(np.int32))
    return holes


def _nearest_gray_labels(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    best_dist = np.full(values.shape, np.inf, dtype=np.float32)
    labels = np.zeros(values.shape, dtype=np.int16)
    for idx, center in enumerate(centers):
        dist = np.abs(values - center)
        take = dist < best_dist
        labels[take] = idx
        best_dist[take] = dist[take]
    return labels


def _patch_rects_from_mask(
    mask: np.ndarray,
    shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    if not mask.any():
        return []
    h, w = shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    work = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    rects: list[tuple[int, int, int, int]] = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 16:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        x0 = max(0, x - _VEC_PATCH_PAD_PX)
        y0 = max(0, y - _VEC_PATCH_PAD_PX)
        x1 = min(w, x + ww + _VEC_PATCH_PAD_PX)
        y1 = min(h, y + hh + _VEC_PATCH_PAD_PX)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            rects.append((x0, y0, x1, y1))
    return rects


def _rect_to_contour(rect: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return np.array(
        [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
        dtype=np.int32,
    )


def _scale_simplified_contour(
    contour: np.ndarray,
    scale: float,
    full_w: int,
    full_h: int,
) -> np.ndarray:
    epsilon = max(0.75, _APPROX_FRAC * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).astype(np.float32)
    if scale > 0:
        approx[:, 0, 0] /= float(scale)
        approx[:, 0, 1] /= float(scale)
    approx[:, 0, 0] = np.clip(np.round(approx[:, 0, 0]), 0, max(0, full_w - 1))
    approx[:, 0, 1] = np.clip(np.round(approx[:, 0, 1]), 0, max(0, full_h - 1))
    return approx.astype(np.int32)


def _bright_knockout_hole_contours(
    gray: np.ndarray,
    valid_mask: np.ndarray,
) -> list[np.ndarray]:
    bright = ((gray >= 245) & valid_mask).astype(np.uint8)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    holes: list[np.ndarray] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < _VEC_FLAT_BRIGHT_HOLE_MIN_AREA_PX:
            continue
        epsilon = max(0.75, _APPROX_FRAC * cv2.arcLength(cnt, True))
        holes.append(cv2.approxPolyDP(cnt, epsilon, True).astype(np.int32))
    return holes


def _holes_inside_mask(
    holes: list[np.ndarray],
    mask: np.ndarray,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    h, w = mask.shape[:2]
    for hole in holes:
        m = cv2.moments(hole)
        if abs(m["m00"]) > 1e-6:
            cx = int(round(m["m10"] / m["m00"]))
            cy = int(round(m["m01"] / m["m00"]))
        else:
            pts = hole[:, 0, :]
            cx = int(round(float(np.mean(pts[:, 0]))))
            cy = int(round(float(np.mean(pts[:, 1]))))
        if 0 <= cx < w and 0 <= cy < h and bool(mask[cy, cx]):
            out.append(hole.copy())
    return out


def _flat_component_distribution_ok(gray: np.ndarray, mask: np.ndarray) -> bool:
    ys, xs = np.where(mask)
    if ys.size < _VEC_FLAT_MIN_COMPONENT_AREA_PX:
        return False
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    medians: list[float] = []
    for yy in range(y0, y1, _VEC_FLAT_TILE_PX):
        for xx in range(x0, x1, _VEC_FLAT_TILE_PX):
            tile_mask = mask[yy:min(y1, yy + _VEC_FLAT_TILE_PX),
                             xx:min(x1, xx + _VEC_FLAT_TILE_PX)]
            if int(tile_mask.sum()) < 16:
                continue
            tile = gray[yy:min(y1, yy + _VEC_FLAT_TILE_PX),
                        xx:min(x1, xx + _VEC_FLAT_TILE_PX)]
            vals = tile[tile_mask]
            vals = vals[vals < 255]
            if vals.size >= 16:
                medians.append(float(np.median(vals)))
    if len(medians) < 4:
        return True
    q25, q75 = np.percentile(np.asarray(medians, dtype=np.float32), [25, 75])
    return float(q75 - q25) <= _VEC_FLAT_TILE_MEDIAN_IQR_MAX
