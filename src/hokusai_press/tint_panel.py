"""Detect flat-tint (平網) panels directly from a page raster.

Unlike _figure_vecfills this detector does not rely on OCR regions.
It works purely from the rendered image by finding rectangular zones
that have a uniform mid-tone gray level -- the signature of a halftone
or solid tint background panel.

Algorithm (dual projection):

Row-projection pass (horizontal bands):
  1. For every row compute the fraction of pixels in [TINT_LO, TINT_HI].
     Rows where this exceeds TINT_PROJ_FRAC are "tint rows".
  2. 1-D closing to bridge short non-tint gaps (text lines within a band).
  3. Extract row runs; find column extent for each run.

Column-projection pass (vertical bands):
  Symmetric to the row pass: "tint columns", column runs, row extent.

Quality filter on every candidate bounding rect (both passes):
  - Minimum area in the projection axis and the cross axis.
  - Low-std block fraction >= LOW_STD_FRAC_MIN to reject random noise / photos.
  - Tone = median of original crop / 255.
"""

from __future__ import annotations

import cv2
import numpy as np

TINT_LO = 60
TINT_HI = 220
TINT_PROJ_FRAC = 0.50       # a row/col is "tint" if >= 50% of its pixels are in range
CLOSE_GAP = 10              # max non-tint rows/cols to bridge
MIN_SPAN_FRAC = 0.030       # run span must be >= 3% of its own axis
MIN_CROSS_FRAC = 0.100      # cross-axis extent must be >= 10% of cross axis
STD_BLOCK_FRAC = 0.010      # block size for local-std check
STD_MAX = 30.0
LOW_STD_FRAC_MIN = 0.25     # >= 25% of blocks must have std < STD_MAX
TINT_IQR_MAX = 65.0         # IQR of in-range pixels must be < this (rejects gradients/photos)


def detect_tint_panels(img: np.ndarray) -> list[dict]:
    """Return flat-tint panel fills detected directly from a page image.

    Returns a list of {"rect": (x0, y0, x1, y1), "gray": float} dicts
    in the same format expected by mrc.MrcPageBuilder.add_page vector_fills.
    """
    if img.size == 0:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = gray.shape[:2]
    tint_px = (gray >= TINT_LO) & (gray <= TINT_HI)

    # Block std map (shared between passes).
    blk = max(4, int(h * STD_BLOCK_FRAC))
    std_map = _block_std_map(gray, blk)
    rows_b = max(1, h // blk)
    cols_b = max(1, w // blk)

    fills: list[dict] = []

    # --- Pass 1: row-projection → horizontal bands ---
    row_frac = tint_px.mean(axis=1)
    row_is_tint = row_frac >= TINT_PROJ_FRAC
    closed_rows = _close_1d(row_is_tint, CLOSE_GAP)
    min_h = max(1, int(h * MIN_SPAN_FRAC))
    min_w = max(1, int(w * MIN_CROSS_FRAC))
    for y0, y1 in _runs(closed_rows, h):
        if y1 - y0 < min_h:
            continue
        col_frac = tint_px[y0:y1, :].mean(axis=0)
        xs = np.where(col_frac >= TINT_PROJ_FRAC)[0]
        if not xs.size or xs[-1] - xs[0] + 1 < min_w:
            continue
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        fill = _make_fill(gray, std_map, rows_b, cols_b, blk, x0, y0, x1, y1)
        if fill is not None:
            fills.append(fill)

    # --- Pass 2: column-projection → vertical bands ---
    col_frac_total = tint_px.mean(axis=0)
    col_is_tint = col_frac_total >= TINT_PROJ_FRAC
    closed_cols = _close_1d(col_is_tint, CLOSE_GAP)
    min_cw = max(1, int(w * MIN_SPAN_FRAC))
    min_ch = max(1, int(h * MIN_CROSS_FRAC))
    for x0, x1 in _runs(closed_cols, w):
        if x1 - x0 < min_cw:
            continue
        row_frac_range = tint_px[:, x0:x1].mean(axis=1)
        ys = np.where(row_frac_range >= TINT_PROJ_FRAC)[0]
        if not ys.size or ys[-1] - ys[0] + 1 < min_ch:
            continue
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        # skip if already substantially covered by a row-pass result
        if _overlaps_any((x0, y0, x1, y1), fills):
            continue
        fill = _make_fill(gray, std_map, rows_b, cols_b, blk, x0, y0, x1, y1)
        if fill is not None:
            fills.append(fill)

    return fills


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_fill(
    gray: np.ndarray,
    std_map: np.ndarray,
    rows_b: int,
    cols_b: int,
    blk: int,
    x0: int, y0: int, x1: int, y1: int,
) -> dict | None:
    """Return {"rect":..., "gray":...} if the bounding rect passes quality
    filters, else None."""
    r0 = y0 // blk; r1 = min(rows_b, (y1 + blk - 1) // blk)
    c0 = x0 // blk; c1 = min(cols_b, (x1 + blk - 1) // blk)
    sub = std_map[r0:r1, c0:c1]
    if sub.size == 0 or float((sub < STD_MAX).mean()) < LOW_STD_FRAC_MIN:
        return None
    crop = gray[y0:y1, x0:x1]
    tint_pix = crop[(crop >= TINT_LO) & (crop <= TINT_HI)]
    if tint_pix.size < 4:
        return None
    q25, q75 = np.percentile(tint_pix, [25, 75])
    if float(q75 - q25) > TINT_IQR_MAX:
        return None
    tone = float(np.median(crop)) / 255.0
    if not (TINT_LO / 255.0 <= tone <= TINT_HI / 255.0):
        return None
    return {"rect": (x0, y0, x1, y1), "gray": tone}


def _overlaps_any(rect: tuple, fills: list[dict]) -> bool:
    """True if rect shares >50% of its area with any existing fill rect."""
    x0, y0, x1, y1 = rect
    area = max(1, (x1 - x0) * (y1 - y0))
    for f in fills:
        fx0, fy0, fx1, fy1 = f["rect"]
        ix = max(0, min(x1, fx1) - max(x0, fx0))
        iy = max(0, min(y1, fy1) - max(y0, fy0))
        if ix * iy > area * 0.5:
            return True
    return False


def _close_1d(mask: np.ndarray, gap: int) -> np.ndarray:
    """Fill False runs of length <= gap bounded by True on both sides."""
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        if j - i <= gap and i > 0 and j < n:
            out[i:j] = True
        i = j
    return out


def _runs(mask: np.ndarray, length: int) -> list[tuple[int, int]]:
    """Return (start, end) pairs of True runs."""
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start = i; in_run = True
        elif not v and in_run:
            runs.append((start, i)); in_run = False
    if in_run:
        runs.append((start, length))
    return runs


def _block_std_map(gray: np.ndarray, blk: int) -> np.ndarray:
    """2-D array of local std on non-overlapping blk×blk blocks."""
    h, w = gray.shape[:2]
    rows_b = max(1, h // blk)
    cols_b = max(1, w // blk)
    std_map = np.zeros((rows_b, cols_b), dtype=np.float32)
    for r in range(rows_b):
        y0b = r * blk; y1b = min(h, y0b + blk)
        for c in range(cols_b):
            x0b = c * blk; x1b = min(w, x0b + blk)
            std_map[r, c] = float(np.std(gray[y0b:y1b, x0b:x1b]))
    return std_map
