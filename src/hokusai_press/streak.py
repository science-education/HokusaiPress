"""Scanner streaks: a line the whole length of the scan, from dust on the
scanner glass or sensor.

A sheet-fed scanner reads each page past a fixed line sensor, so a speck on
the glass draws the same thin line along the feed direction through
everything -- paper, print, the gray bed -- from the first scanned row to the
last. On img20261121_0014 the duplex back sensor drew one about 8 px wide
(on paper 245 -> 186 at its core; brighter on the gray bed) on the even pages
from p208 to p274. It is the straightest line on the page and can outvote
the text in deskew: p274, laid 6 deg askew, stayed at 0 deg; with the
streak painted out it deskews to 6.2 deg.

There is no telling what a streak hides, so it is reported, not repaired:
the page is flagged SCANNER_STREAK and the run warns to clean the scanner.

Detection (per page, along the feed axis): a column stands out from both
sides alike (a line, not the step at a paper edge) over most of the scanned
length AND within both its first and last 5% -- printed rules stop at the
margins, a streak runs to the ends of the scan -- AND, where it crosses
paper, keeps one value row after row (one sensor element): a rule printed
from edge to edge (a full-bleed ornament, img20261121_0007 p9) varies with
the ink.
"""

from __future__ import annotations

import cv2
import numpy as np

STREAK_MIN_DEV = 10        # gray levels the line differs from both sides
STREAK_MIN_COVER = 0.8     # ... over this share of the scanned length
STREAK_END_COVER = 0.9     # ... and of its first and last STREAK_END_FRAC
STREAK_END_FRAC = 0.05
STREAK_MAX_WIDTH = 16      # px at 300 dpi (measured: ~8)
STREAK_SIDE = 12           # px from the line to the sides it is compared with
STREAK_EDGE_SKIP = 16      # px next to the image (or pad) edge are not looked at
STREAK_MIN_PAPER = 0.2     # it crosses paper (both sides paper) over this share ...
STREAK_MAX_JITTER = 8.0    # ... where its core keeps one value: IQR (streak 2-5, print 11-29)


def find_streaks(gray: np.ndarray, axis: int = 0) -> list[tuple[float, int]]:
    """Streaks running along `axis` (0: vertical lines, the feed direction of
    an upright page; 1: horizontal) as [(position as a fraction of the image
    width/height, width in px)]. `gray`: the page at about 300 dpi."""
    from .source import _white_pad

    g = gray.astype(np.float32)
    if axis == 1:
        g = g.T
    x0, x1 = _white_pad(g, 0)              # the scanner's pad is not scanned
    y0, y1 = _white_pad(g, 1)
    full_w = g.shape[1]
    g = g[y0:y1, x0:x1]
    h, w = g.shape
    m, s = STREAK_EDGE_SKIP, STREAK_SIDE
    if h < 100 or w <= 2 * (m + s):
        return []
    core = cv2.blur(g, (3, 1))
    side = cv2.blur(g, (7, 1))
    left = np.zeros_like(g)
    right = np.zeros_like(g)
    left[:, s:] = side[:, :-s]
    right[:, :-s] = side[:, s:]
    dl, dr = core - left, core - right
    dev = np.where(np.sign(dl) == np.sign(dr), np.minimum(np.abs(dl), np.abs(dr)), 0.0)
    on = dev > STREAK_MIN_DEV
    cover = on.mean(axis=0)
    cover[:m] = 0.0
    cover[w - m:] = 0.0
    xs = np.flatnonzero(cover >= STREAK_MIN_COVER)
    if xs.size == 0:
        return []
    e = max(1, int(STREAK_END_FRAC * h))
    level = float(np.percentile(g[::4, ::4], 90)) - 40.0   # paper, whatever its tone
    out = []
    for run in np.split(xs, np.flatnonzero(np.diff(xs) > 2) + 1):
        width = int(run[-1] - run[0] + 1)
        if width > STREAK_MAX_WIDTH:
            continue
        line = on[:, run].any(axis=1)
        if line[:e].mean() < STREAK_END_COVER or line[-e:].mean() < STREAK_END_COVER:
            continue
        if not _steady_on_paper(g, int(run[0]), int(run[-1]), level):
            continue
        out.append((float((x0 + (run[0] + run[-1]) / 2.0) / full_w), width))
    return out


def _steady_on_paper(g: np.ndarray, a: int, b: int, level: float) -> bool:
    """Where paper lies on both sides, a sensor streak darkens it by the same
    amount on every row (one sensor element); a printed rule varies with the
    ink (img20261121_0007 p9: IQR 23-29 against 2-5 for 0014's streak)."""
    h, w = g.shape
    lo, hi = max(0, a - 2), min(w, b + 3)
    core = g[:, lo:hi].min(axis=1)
    sl = g[:, max(0, a - 2 * STREAK_SIDE):max(1, a - STREAK_SIDE)]
    sr = g[:, min(w - 1, b + STREAK_SIDE):min(w, b + 2 * STREAK_SIDE)]
    if sl.size == 0 or sr.size == 0:
        return False
    paper = (sl.mean(axis=1) > level) & (sr.mean(axis=1) > level)
    if paper.mean() < STREAK_MIN_PAPER:
        return False
    q75, q25 = np.percentile(core[paper], [75, 25])
    return float(q75 - q25) <= STREAK_MAX_JITTER


def page_streaks(img: np.ndarray, rotate: int = 0) -> list[tuple[float, int]]:
    """find_streaks along the scanner's feed axis of a page as the pipeline
    sees it: vertical in the scan, horizontal once turned 90/270 deg."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return find_streaks(gray, axis=1 if int(rotate or 0) % 180 else 0)


def _ranges(nums: list[int]) -> str:
    """'207-265 every 2nd, 273' -- runs with a step of 1 or 2 collapsed."""
    nums = sorted(nums)
    parts, i = [], 0
    while i < len(nums):
        j = i
        step = nums[i + 1] - nums[i] if i + 1 < len(nums) and nums[i + 1] - nums[i] in (1, 2) else 0
        while step and j + 1 < len(nums) and nums[j + 1] - nums[j] == step:
            j += 1
        if j - i >= 2:
            parts.append(f"p{nums[i]}-p{nums[j]}" + (" every 2nd" if step == 2 else ""))
            i = j + 1
        else:
            parts.append(f"p{nums[i]}")
            i += 1
    return ", ".join(parts)


def streak_warning(found: dict[int, list[tuple[float, int]]]) -> "str | None":
    """One warning line for a book: {page_index: streaks}, the pages named
    by their PDF page numbers (page_index + 1)."""
    found = {i: s for i, s in found.items() if s}
    if not found:
        return None
    where = sorted({round(pos * 100) for s in found.values() for pos, _w in s})
    return (f"scanner streak on {len(found)} page(s): {_ranges([i + 1 for i in found])} "
            f"(a line the full length of the scan at {', '.join(f'{p}%' for p in where)} "
            f"of the width) -- dust on the scanner glass/sensor; clean it and "
            f"rescan these pages: deskew and OCR can be thrown off")
