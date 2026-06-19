"""Tests for tint_zone module."""

import numpy as np
import pytest

from hokusai_press.tint_zone import (
    TINT_HI,
    TINT_LO,
    TintZone,
    build_tint_zones,
    classify_tint_shape,
    make_tint_overlay,
)


# ---------------------------------------------------------------------------
# make_tint_overlay
# ---------------------------------------------------------------------------

def test_overlay_paper_pixels_become_255():
    """Pixels > TINT_HI must be set to 255 (transparent to Multiply)."""
    crop = np.array([[100, 200, 240, 255]], dtype=np.uint8)
    ov = make_tint_overlay(crop)
    assert ov[0, 0] == 100  # tint pixel unchanged
    assert ov[0, 1] == 200  # tint pixel unchanged
    assert ov[0, 2] == 255  # 240 > TINT_HI → 255
    assert ov[0, 3] == 255  # 255 > TINT_HI → 255


def test_overlay_dark_pixels_unchanged():
    """Pixels < TINT_LO are kept as-is (very rare in practice)."""
    crop = np.array([[30, 50, 60, 170]], dtype=np.uint8)
    ov = make_tint_overlay(crop)
    assert ov[0, 0] == 30
    assert ov[0, 1] == 50
    assert ov[0, 2] == 60   # TINT_LO == 60 → boundary: 60 is IN [TINT_LO, TINT_HI]
    assert ov[0, 3] == 170


def test_overlay_preserves_tint_range():
    rng = np.random.default_rng(0)
    crop = rng.integers(TINT_LO, TINT_HI + 1, (50, 100), dtype=np.uint8)
    ov = make_tint_overlay(crop)
    # All pixels in tint range → no change
    np.testing.assert_array_equal(ov, crop)


# ---------------------------------------------------------------------------
# classify_tint_shape
# ---------------------------------------------------------------------------

def test_classify_full_tint_mask_is_rect():
    mask = np.ones((100, 300), dtype=np.uint8)
    shape, r, pts = classify_tint_shape(mask)
    assert shape == "rect"
    assert r == 0.0


def test_classify_mostly_full_mask_is_rect():
    """A mask with 94% coverage (a few random holes) should still be 'rect'."""
    rng = np.random.default_rng(1)
    mask = (rng.random((100, 200)) > 0.06).astype(np.uint8)
    shape, _, _ = classify_tint_shape(mask)
    assert shape == "rect"


def test_classify_scattered_missing_pixels_stays_rect():
    """Uniformly scattered missing pixels must NOT trigger rounded_rect."""
    rng = np.random.default_rng(42)
    mask = (rng.random((200, 800)) > 0.12).astype(np.uint8)  # ~88% coverage
    # Ensure coverage is between ROUNDED_COVERAGE_MIN and RECT_COVERAGE_MIN
    assert 0.85 <= mask.mean() <= 0.95
    shape, _, _ = classify_tint_shape(mask)
    # Scattered pixels → rect (not rounded_rect), because centroids are centred
    assert shape == "rect"


def test_classify_real_rounded_corner():
    """A mask with genuine corner cutouts should be classified as rounded_rect.

    We use h=w=100, r=25 so that the four cutouts remove ~5.4% of area and
    coverage ≈ 0.946 -- between ROUNDED_COVERAGE_MIN (0.85) and
    RECT_COVERAGE_MIN (0.95), which forces the corner-radius estimation path.
    """
    h, w, r = 100, 100, 25
    mask = np.ones((h, w), dtype=np.uint8)
    corners_setup = [
        (range(r), range(r), r, r),                          # top-left
        (range(r), range(w - r, w), r, w - r),               # top-right
        (range(h - r, h), range(r), h - r, r),               # bottom-left
        (range(h - r, h), range(w - r, w), h - r, w - r),    # bottom-right
    ]
    for ys, xs, cy, cx in corners_setup:
        for y in ys:
            for x in xs:
                if (y - cy) ** 2 + (x - cx) ** 2 > r ** 2:
                    mask[y, x] = 0

    coverage = float(mask.mean())
    assert 0.85 < coverage < 0.95, f"coverage {coverage:.3f} outside expected range"

    shape, est_r, _ = classify_tint_shape(mask)
    assert shape == "rounded_rect", f"got {shape} (coverage={coverage:.3f})"
    assert 20 <= est_r <= 35, f"corner radius estimate {est_r:.1f} out of expected [20, 35]"


# ---------------------------------------------------------------------------
# build_tint_zones integration
# ---------------------------------------------------------------------------

def _make_gray_with_tint(h, w, y0, y1, x0, x1, tint_val=170):
    gray = np.full((h, w), 240, dtype=np.uint8)
    gray[y0:y1, x0:x1] = tint_val
    return gray


def test_build_zones_single_row_panel():
    gray = _make_gray_with_tint(600, 400, 10, 100, 10, 390, 170)
    panels = [{"rect": (10, 10, 390, 100), "gray": 170 / 255}]
    zones = build_tint_zones(gray, panels)
    assert len(zones) == 1
    z = zones[0]
    assert isinstance(z, TintZone)
    assert z.hole_contours == []
    # Contour-based detection: rect may be slightly larger than the panel hint
    # (padding); verify the original panel area is covered.
    assert z.rect[0] <= 10 and z.rect[1] <= 10
    assert z.rect[2] >= 390 and z.rect[3] >= 100
    # Overlay includes paper pixels (expanded crop), so paper → 255.
    # Tint pixels (170 ≤ TINT_HI) remain 170 → min is 170.
    assert z.overlay_img.min() == 170


def test_build_zones_separate_panels_stay_separate():
    """Two tint panels that don't touch must remain two distinct zones."""
    h, w = 1000, 400
    gray = np.full((h, w), 240, dtype=np.uint8)
    gray[0:100, 0:400] = 170     # header
    gray[900:1000, 0:400] = 170  # footer, far from header
    panels = [
        {"rect": (0, 0, 400, 100), "gray": 0.67},
        {"rect": (0, 900, 400, 1000), "gray": 0.67},
    ]
    zones = build_tint_zones(gray, panels)
    assert len(zones) == 2
    assert all(z.hole_contours == [] for z in zones)


def test_build_zones_connected_frame_has_one_hole():
    """A header + footer + two columns that physically touch at their
    corners form a single connected "frame" fully enclosing the body-text
    area on all four sides.  The whole-page hierarchy-aware contour pass
    must return ONE zone with ONE hole covering the body cavity -- not
    several separately-clipped zones stitched together with an arbitrary
    seam.  (A cavity that isn't enclosed on all sides -- e.g. left open to
    the image border -- isn't a true topological hole, so the frame must
    close on every side for cv2's hierarchy to detect it.)
    """
    h, w = 1000, 400
    gray = np.full((h, w), 240, dtype=np.uint8)
    gray[0:100, 0:400] = 170     # header (full width)
    gray[900:1000, 0:400] = 170  # footer (full width)
    gray[0:1000, 0:60] = 170     # left column (full height)
    gray[0:1000, 340:400] = 170  # right column (full height) -- closes the frame
    # body cavity: x=60-340, y=100-900 stays paper (240), enclosed on all sides

    panels = [
        {"rect": (0, 0, 400, 100), "gray": 0.67},
        {"rect": (0, 900, 400, 1000), "gray": 0.67},
        {"rect": (0, 0, 60, 1000), "gray": 0.67},
        {"rect": (340, 0, 400, 1000), "gray": 0.67},
    ]
    zones = build_tint_zones(gray, panels)

    assert len(zones) == 1
    z = zones[0]
    assert z.shape_type == "frame"
    assert len(z.hole_contours) == 1

    hx, hy, hw, hh = __import__("cv2").boundingRect(z.hole_contours[0])
    # Hole should approximate the body cavity (x=60-340, y=100-900).
    assert hx >= 50 and hy >= 90
    assert hx + hw <= 350 and hy + hh <= 910
    assert hw >= 250 and hh >= 750  # most of the cavity is captured
