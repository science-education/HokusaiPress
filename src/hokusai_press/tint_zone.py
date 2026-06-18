"""High-quality tint-zone analysis and overlay preparation.

A "tint zone" is a region of the page that carries a flat or slowly-varying
grey background (平網 / halftone tint) -- usually a header, footer, or article
background panel -- with text or other marks printed on top of it.

The goal is to reproduce this appearance in the output PDF with maximum fidelity:
  - The tint background must be rendered at its actual per-pixel colour, not
    approximated by a single median value.
  - Text strokes on the tint must remain sharp bilevel, not blurred by JPEG.
  - White paper areas within the panel boundary must stay white.

Architecture (layered compositing with Multiply blend):

  1. Bilevel base layer   -- Otsu-binarised; text strokes already black,
                            tint background already white. No modification needed.
  2. Tint overlay (new)  -- Greyscale image placed with /BM /Multiply:
                            • tint pixels  : original grey value  → darkens the
                              white bilevel to the correct tint colour.
                            • paper pixels : 255                  → Multiply(1.0)
                              is identity, so paper stays white.
                            • text pixels  : original grey value  → Multiply(v, 0)
                              = 0 regardless of v; text stays black in bilevel.
                              (Setting text pixels to 255 is optional and only
                              affects JPEG compression, not rendering.)

The overlay image is therefore constructed as:
    overlay = gray_crop.copy()
    overlay[gray_crop > TINT_HI] = 255   # paper transparent to Multiply

Overlap deduplication:
  Row-projection panels and column-projection panels can overlap at the corners.
  A second Multiply pass over the same area would double-darken it.
  Column panels are clipped to remove any y-range already covered by a row panel,
  and then sorted so the caller always receives non-overlapping rectangles.

Shape classification (foundation; rounded-rect detection is wired but deferred):
  "rect"         -- tint coverage ≥ 95 % of the bounding box.
  "rounded_rect" -- ≥ 85 % coverage, corners show circular cutouts.
  "poly"         -- otherwise (arbitrary contour, stored for future use).
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


@dataclass
class TintZone:
    """Rendering data for one tint panel."""

    rect: tuple[int, int, int, int]      # (x0, y0, x1, y1) in output-image pixels
    overlay_img: np.ndarray              # HxW uint8 greyscale; 255 = transparent
    shape_type: str = "rect"             # "rect" | "rounded_rect" | "poly"
    corner_radius: float = 0.0          # pixels, for "rounded_rect"
    contour_pts: list = field(default_factory=list)  # for "poly" (list of (x,y))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tint_zones(
    gray: np.ndarray,
    panels: list[dict],
) -> list[TintZone]:
    """Convert raw detected panel dicts to TintZone objects.

    ``panels`` is the output of ``tint_panel.detect_tint_panels``.
    Steps:
      1. Separate panels into row-pass results (wide, short) and column-pass
         results (narrow, tall) by aspect ratio.
      2. Clip column panels to remove y-ranges covered by row panels.
      3. For each surviving rect, call ``_make_tint_zone``.

    Returns a list of non-overlapping TintZone objects ready for MRC rendering.
    """
    if not panels:
        return []

    h, w = gray.shape[:2]
    row_panels, col_panels = _split_by_orientation(panels, w, h)

    # Build row zones (no modification needed – they take priority).
    row_zones: list[TintZone] = []
    for p in row_panels:
        z = _make_tint_zone(gray, p["rect"])
        if z is not None:
            row_zones.append(z)

    # Clip column panels: remove y-ranges already covered by a row panel.
    col_zones: list[TintZone] = []
    row_y_intervals = [(z.rect[1], z.rect[3]) for z in row_zones]
    for p in col_panels:
        for clipped_rect in _clip_col_panel(p["rect"], row_y_intervals, h):
            z = _make_tint_zone(gray, clipped_rect)
            if z is not None:
                col_zones.append(z)

    return row_zones + col_zones


def make_tint_overlay(gray_crop: np.ndarray) -> np.ndarray:
    """Return the overlay image for a tint panel crop.

    Paper pixels (> TINT_HI) are set to 255 so that a Multiply-blend overlay
    has no effect on those areas.  All other pixels keep their original grey
    value.  The overlay is rendered with /BM /Multiply in the PDF, so:
      - tint background (grey):  Multiply(v/255, 1.0) = v/255  → tint shows ✓
      - paper white   (255):     Multiply(1.0,    1.0) = 1.0    → stays white ✓
      - on text stroke (bilevel=0): Multiply(v/255, 0.0) = 0   → stays black ✓
    """
    overlay = gray_crop.copy()
    overlay[gray_crop > TINT_HI] = 255
    return overlay


def classify_tint_shape(
    tint_mask: np.ndarray,
) -> tuple[str, float, list]:
    """Classify the shape of a tint panel from its binary mask.

    Returns (shape_type, corner_radius_px, contour_pts).
    corner_radius_px is meaningful only for "rounded_rect".
    contour_pts is a list of (x,y) ints for "poly".
    """
    h, w = tint_mask.shape[:2]
    total = h * w
    coverage = float(tint_mask.mean())

    if coverage >= RECT_COVERAGE_MIN:
        return "rect", 0.0, []

    # Find the outer contour.
    contours, _ = cv2.findContours(
        tint_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return "rect", 0.0, []
    cnt = max(contours, key=cv2.contourArea)

    if coverage < ROUNDED_COVERAGE_MIN:
        pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
        return "poly", 0.0, pts

    # Attempt rounded-rect: check the four corner quadrants for circular cutouts.
    r = _estimate_corner_radius(tint_mask, cnt)
    min_r = max(1.0, min(h, w) * CORNER_RADIUS_MIN_FRAC)
    if r >= min_r:
        return "rounded_rect", r, []

    return "rect", 0.0, []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_tint_zone(
    gray: np.ndarray,
    rect: tuple[int, int, int, int],
) -> TintZone | None:
    """Build a TintZone for one bounding rect."""
    x0, y0, x1, y1 = rect
    h_full, w_full = gray.shape[:2]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w_full, x1); y1 = min(h_full, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    gray_crop = gray[y0:y1, x0:x1]
    overlay = make_tint_overlay(gray_crop)

    tint_mask = ((gray_crop >= TINT_LO) & (gray_crop <= TINT_HI)).astype(np.uint8)
    shape_type, corner_radius, contour_pts = classify_tint_shape(tint_mask)

    return TintZone(
        rect=(x0, y0, x1, y1),
        overlay_img=overlay,
        shape_type=shape_type,
        corner_radius=corner_radius,
        contour_pts=contour_pts,
    )


def _split_by_orientation(
    panels: list[dict],
    page_w: int,
    page_h: int,
) -> tuple[list[dict], list[dict]]:
    """Separate panels into row-pass (horizontal bands) and column-pass (vertical bands).

    Heuristic: a panel whose width/height ratio >= 2.0 is a horizontal band;
    a panel whose height/width ratio >= 2.0 is a vertical band.  Panels
    close to square are added to both for safety (rare in practice).
    """
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
            row_panels.append(p)   # square-ish → treat as row (conservative)
    return row_panels, col_panels


def _clip_col_panel(
    col_rect: tuple[int, int, int, int],
    row_y_intervals: list[tuple[int, int]],
    page_h: int,
) -> list[tuple[int, int, int, int]]:
    """Remove the y-ranges covered by row panels from a column panel rect.

    Returns 0, 1, or 2 non-overlapping rectangles (same x0/x1, clipped y).
    """
    x0, y0, y1_orig = col_rect[0], col_rect[1], col_rect[3]
    x1 = col_rect[2]

    # Build a sorted list of "blocked" y intervals.
    blocked: list[tuple[int, int]] = []
    for ry0, ry1 in row_y_intervals:
        lo = max(y0, ry0); hi = min(y1_orig, ry1)
        if hi > lo:
            blocked.append((lo, hi))
    if not blocked:
        return [(x0, y0, x1, y1_orig)]

    blocked.sort()
    # Merge overlapping/touching intervals.
    merged: list[tuple[int, int]] = [blocked[0]]
    for lo, hi in blocked[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    # Build free intervals.
    free: list[tuple[int, int, int, int]] = []
    cursor = y0
    for blo, bhi in merged:
        if cursor < blo:
            free.append((x0, cursor, x1, blo))
        cursor = max(cursor, bhi)
    if cursor < y1_orig:
        free.append((x0, cursor, x1, y1_orig))

    # Keep only intervals that meet minimum height.
    min_h = max(4, int(page_h * 0.030))
    return [(x0, y0c, x1, y1c) for (x0, y0c, x1, y1c) in free if y1c - y0c >= min_h]


def _estimate_corner_radius(
    tint_mask: np.ndarray,
    contour: np.ndarray,
) -> float:
    """Estimate the average corner radius from the four corner regions of the mask.

    A genuine rounded corner has its missing tint pixels concentrated near the
    geometric corner of the quadrant.  Scattered white pixels (from text strokes,
    paper show-through, or ADF scan variation) have their centroid near the
    quadrant centre -- these are rejected.

    The concentration test: the centroid of missing pixels must be within the
    outer 35% of the quadrant (measured as a fraction of the diagonal from
    centre to corner).  Anything further in is considered uniformly scattered.
    """
    h, w = tint_mask.shape[:2]
    qh = max(1, h // 5)
    qw = max(1, w // 5)
    radii: list[float] = []
    # corner_id is used to determine which direction is "toward the corner"
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
        # Normalised centroid in [0,1]×[0,1] for this quadrant
        cy_frac = float(missing_ys.mean()) / max(1, rh - 1)  # 0 = top, 1 = bottom
        cx_frac = float(missing_xs.mean()) / max(1, rw - 1)  # 0 = left, 1 = right

        # For a true rounded corner the centroid should be TOWARD the geometric corner.
        # For "tl" the corner direction is (0,0); for "br" it is (1,1), etc.
        corner_cx, corner_cy = {
            "tl": (0.0, 0.0), "tr": (1.0, 0.0),
            "bl": (0.0, 1.0), "br": (1.0, 1.0),
        }[cid]
        dist_from_corner = ((cx_frac - corner_cx) ** 2 + (cy_frac - corner_cy) ** 2) ** 0.5
        # Maximum distance is ~1.41 (diagonal); a centroid within 0.45 of the
        # corner means it is concentrated in roughly the outer 30% of the diagonal.
        if dist_from_corner > 0.45:
            continue   # scattered pixels, not a real rounded corner

        missing = int(missing_ys.size)
        # A rounded corner removes the area between the rectangle corner and
        # the arc: missing ≈ r² - π*r²/4 = r² * (1 - π/4) ≈ 0.2146 * r²
        # Solve for r:
        r = float(np.sqrt(missing / (1.0 - np.pi / 4.0)))
        # Corner radius can be at most half the short side of the whole panel.
        if r <= min(h, w) / 2:
            radii.append(r)
    return float(np.mean(radii)) if radii else 0.0
