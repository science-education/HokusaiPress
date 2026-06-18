"""Tests for tint_zone module."""

import numpy as np
import pytest

from hokusai_press.tint_zone import (
    TINT_HI,
    TINT_LO,
    TintZone,
    _clip_col_panel,
    _split_by_orientation,
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
# _split_by_orientation
# ---------------------------------------------------------------------------

def test_split_wide_panel_is_row():
    panels = [{"rect": (0, 0, 800, 100), "gray": 0.7}]  # w/h = 8 → row
    row_panels, col_panels = _split_by_orientation(panels, 800, 1200)
    assert len(row_panels) == 1
    assert len(col_panels) == 0


def test_split_narrow_panel_is_col():
    panels = [{"rect": (0, 0, 80, 1000), "gray": 0.7}]  # h/w = 12.5 → col
    row_panels, col_panels = _split_by_orientation(panels, 800, 1200)
    assert len(row_panels) == 0
    assert len(col_panels) == 1


# ---------------------------------------------------------------------------
# _clip_col_panel
# ---------------------------------------------------------------------------

def test_clip_removes_row_overlap():
    col = (10, 0, 50, 1000)  # x=10-50, y=0-1000
    row_intervals = [(0, 200), (800, 1000)]
    result = _clip_col_panel(col, row_intervals, 1000)
    # Remaining free: y=200-800
    assert len(result) == 1
    x0, y0, x1, y1 = result[0]
    assert x0 == 10 and x1 == 50
    assert y0 == 200 and y1 == 800


def test_clip_no_overlap_returns_original():
    col = (10, 500, 50, 800)
    row_intervals = [(0, 100), (900, 1000)]
    result = _clip_col_panel(col, row_intervals, 1000)
    assert len(result) == 1
    assert result[0] == col


def test_clip_fully_covered_returns_empty():
    col = (10, 100, 50, 200)
    row_intervals = [(0, 1000)]
    result = _clip_col_panel(col, row_intervals, 1000)
    assert result == []


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
    assert z.rect == (10, 10, 390, 100)
    # Crop is entirely tint (170); no pixels > TINT_HI, so overlay is unchanged
    assert z.overlay_img.max() == 170
    assert z.overlay_img.min() == 170


def test_build_zones_column_clipped_by_row():
    """Column panel that overlaps at both ends with row panels should be clipped."""
    h, w = 1000, 400
    # Row panels occupy y=0-100 and y=900-1000
    gray = np.full((h, w), 240, dtype=np.uint8)
    gray[0:100, 5:50] = 170    # col + row overlap top
    gray[100:900, 5:50] = 170  # col body
    gray[900:1000, 5:50] = 170 # col + row overlap bottom
    row_p = [
        {"rect": (0, 0, 400, 100), "gray": 0.67},
        {"rect": (0, 900, 400, 1000), "gray": 0.67},
    ]
    col_p = [{"rect": (5, 0, 50, 1000), "gray": 0.67}]
    zones = build_tint_zones(gray, row_p + col_p)
    # 2 row zones + 1 clipped col zone (y=100-900)
    rects = [z.rect for z in zones]
    col_zone = [z for z in zones if z.rect[2] - z.rect[0] < z.rect[3] - z.rect[1]]
    assert len(col_zone) == 1
    assert col_zone[0].rect[1] >= 100   # clipped away the row-zone y-range
    assert col_zone[0].rect[3] <= 900
