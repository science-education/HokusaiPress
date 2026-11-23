import cv2
import numpy as np

from hokusai_press.streak import _ranges, find_streaks, page_streaks, streak_warning


def _page(h=1500, w=1100, pad=31):
    """A white page of short text lines inside the scanner's white pad."""
    g = np.full((h, w), 246, np.uint8)
    for y in range(200, h - 200, 40):
        for x in range(150, w - 150, 30):
            cv2.rectangle(g, (x, y), (x + 20, y + 20), 30, 1)
    return g, pad


def _padded(g, pad):
    return cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def _streak(g, x, depth=55, width=6):
    """img20261121_0014 p274: ~8 px, 245 -> 186 at the core, the whole length."""
    g = g.astype(np.int16)
    g[:, x:x + width] -= depth
    return np.clip(g, 0, 255).astype(np.uint8)


def test_streak_the_full_length_of_the_scan_is_found():
    g, pad = _page()
    img = _padded(_streak(g, 1000), pad)   # in the margin, like 0014's at 94%
    found = find_streaks(img)
    assert len(found) == 1
    pos, width = found[0]
    assert abs(pos * img.shape[1] - (pad + 1003)) < 3 and width <= 8


def test_printed_rule_stopping_at_the_margins_is_no_streak():
    g, pad = _page()
    g[150:-150, 900:906] = 40                  # a rule in the type area only
    assert find_streaks(_padded(g, pad)) == []


def test_paper_edge_against_the_bed_is_no_streak():
    g, pad = _page()
    g[:, 950:] = 123                           # the gray bed beside the sheet
    assert find_streaks(_padded(g, pad)) == []


def test_clean_page_has_no_streak():
    g, pad = _page()
    assert find_streaks(_padded(g, pad)) == []


def test_streak_runs_along_the_feed_of_a_turned_page():
    g, pad = _page()
    img = cv2.rotate(_padded(_streak(g, 1000), pad), cv2.ROTATE_90_CLOCKWISE)
    assert page_streaks(img, rotate=0) == []          # across the feed: nothing
    assert len(page_streaks(img, rotate=90)) == 1     # turned page: along rows


def test_warning_lists_pages_compactly():
    assert _ranges([207, 209, 211, 213, 273]) == "p207-p213 every 2nd, p273"
    assert _ranges([1, 2, 3, 7, 9]) == "p1-p3, p7, p9"
    w = streak_warning({207: [(0.94, 8)], 209: [(0.94, 8)], 211: [(0.941, 7)], 5: []})
    assert w.startswith("scanner streak on 3 page(s): p207-p211 every 2nd")
    assert "94%" in w
    assert streak_warning({1: [], 2: []}) is None


def test_full_bleed_printed_rule_is_no_streak():
    # img20261121_0007 p9: ornamental rules printed edge to edge; the ink
    # varies row to row, a sensor streak does not
    g, pad = _page()
    rng = np.random.default_rng(0)
    g[:, 900:906] = rng.integers(60, 140, size=(g.shape[0], 1))
    assert find_streaks(_padded(g, pad)) == []
