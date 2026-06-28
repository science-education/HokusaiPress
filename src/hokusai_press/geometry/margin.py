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

from ..model import Box, Deskew, Flag, Margin, RegionKind

SPECKLE_AREA_FRAC = 1e-5     # components smaller than this fraction are noise
NOMBRE_BAND_FRAC = 0.08      # top/bottom 8% of the page is the nombre band
NOMBRE_MAX_HEIGHT_FRAC = 0.04


SHADOW_EDGE_FRAC = 0.15      # a shadow lives within the outer 15% of a side
SHADOW_SIDE_FRAC = 0.5       # ... and runs >= half that side's length
SHADOW_THIN_FRAC = 0.15      # ... while staying thin (not a big figure)

# A binding/ADF shadow doesn't always run a clean >=50% of the side -- partial
# contact during the scan can leave it covering as little as ~15% (measured
# on real corpus pages: img20260427_0001 has a 1-6px line hugging the left
# edge at the SAME x across many pages, 15.6%-49.5% of page height). The
# SHADOW_SIDE_FRAC rule above misses these because requiring only "thin" at
# 15% width still risks eating real content. The fix is tightening the OTHER
# axis instead: within the truly extreme edge (not just outer 15%, but outer
# ~3% -- the scanner-bleed zone no real text ever starts in) a thin component
# is a shadow fragment regardless of how much of the side it covers.
EXTREME_EDGE_FRAC = 0.03
EXTREME_THIN_FRAC = 0.01
EXTREME_SIDE_FRAC = 0.12     # still need *some* length, to spare true speckle

# Binarization threshold for the bw output layer, shadow detection and the blank
# gate. Otsu finds the real ink/paper valley on a normal scan and must be used as
# is: on this ADF the text strokes span the ~130-218 gray band (they are NOT all
# darker than ~100), so the valley typically lands at 180-218. A fixed ceiling
# (the old min(Otsu, 130)) sat *below* that valley and dropped 38-53% of the ink
# on content pages -> faint, broken characters on every page, and it also blinded
# remove_edge_shadows to mid-gray shadow bands. Otsu is only untrustworthy in one
# regime: a near-blank page carrying nothing but show-through (裏移り) / shadow
# penumbra has no dark ink mode, so Otsu has nothing to lock onto and shoots up to
# ~250, which would turn that show-through black. Detect *that* regime by its
# signature -- a high Otsu AND essentially no genuinely dark pixels -- and only
# then clamp to INK_FLOOR so show-through is rejected and the blank gate can
# retire the page. Measured over 970 real pages: every inked page has Otsu <= 218
# with >= 1% of pixels darker than INK_FLOOR; the two show-through pages have
# Otsu ~250 with 0.000% that dark -- a clean, two-signal separation.
INK_VALLEY_MAX = 225        # Otsu above this *may* be a degenerate near-blank page
INK_FLOOR = 110             # ... confirmed if <0.1% of pixels are this dark; then
DARK_INK_MIN_FRAC = 0.001   #     only genuinely dark pixels count as ink
# When a document-wide valley is known (2-pass), a show-through page is one
# whose own Otsu sits well ABOVE the book's valley. The book's per-page Otsu is
# very stable (measured on tmp0613: std 2-9 within a book), so the book median
# is a robust valley and "own Otsu > book_valley + delta" is a cleaner
# show-through test than the standalone darkfrac magic below.
INK_SHOWTHROUGH_DELTA = 25


def document_ink_valley(grays) -> "int | None":
    """Robust per-document binarization valley: the median of each page's Otsu
    over an iterable of page grays. Used by the 2-pass binarization so a
    show-through page (no dark mode -> Otsu shoots high) can fall back to the
    book's normal valley instead of a fixed floor. None if no pages."""
    otsus = []
    for g in grays:
        if g is None or g.ndim != 2 or not g.size:
            continue
        o, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsus.append(int(o))
    if not otsus:
        return None
    return int(np.median(otsus))


def ink_threshold(gray: np.ndarray, book_valley: "int | None" = None) -> int:
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = int(otsu)
    if book_valley is not None:
        # 2-pass: a page whose Otsu sits well above the book's stable valley has
        # no real dark ink (show-through / blank) -> use the book valley so its
        # bleed-through isn't turned black. Normal pages keep their own Otsu.
        if otsu > book_valley + INK_SHOWTHROUGH_DELTA:
            return book_valley
        return otsu
    # standalone fallback (no document context): the original two-signal test.
    if otsu > INK_VALLEY_MAX and float((gray <= INK_FLOOR).mean()) < DARK_INK_MIN_FRAC:
        return INK_FLOOR
    return otsu


def remove_edge_shadows(img: np.ndarray) -> np.ndarray:
    """Whiten dark binding/ADF shadow bands along the page edges (gray or BGR).

    Connected-component rule (deterministic, no ML): a single dark component that
    lives in the outer 15% of a side, runs for >= half that side, and stays thin
    is a scan shadow -> set its pixels white. Component-based (not whole-column)
    so a slanted/curved shadow is removed in full, not left as a triangular
    remnant. A real text line is many short components, never one long bar, so
    content is preserved. Otsu makes it catch soft gray shadows too.

    Used once on the original before everything (content detection, deskew,
    render) so a shadow can neither be mistaken for content nor survive into the
    output.
    """
    # Remove fragmented binding lines FIRST, from the pristine ink: the shadow
    # rule below would otherwise eat a line's solid core and leave short dashes
    # whose vertical extent no longer reads as a line (near-blank pages only).
    img = _remove_fragmented_edge_lines(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    # absolute-floored "ink" mask (not raw Otsu): on a near-blank page this keeps
    # only the genuinely dark edge line as a component, so it is whitened cleanly
    # instead of drowning in show-through noise.
    binv = (gray <= ink_threshold(gray)).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    ew, eh = w * SHADOW_EDGE_FRAC, h * SHADOW_EDGE_FRAC
    xew, xeh = w * EXTREME_EDGE_FRAC, h * EXTREME_EDGE_FRAC
    out = img.copy()
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        v = (ch >= SHADOW_SIDE_FRAC * h and cw <= SHADOW_THIN_FRAC * w
             and (x <= ew or x + cw >= w - ew))
        hsh = (cw >= SHADOW_SIDE_FRAC * w and ch <= SHADOW_THIN_FRAC * h
               and (y <= eh or y + ch >= h - eh))
        xv = (ch >= EXTREME_SIDE_FRAC * h and cw <= EXTREME_THIN_FRAC * w
              and (x <= xew or x + cw >= w - xew))
        xh = (cw >= EXTREME_SIDE_FRAC * w and ch <= EXTREME_THIN_FRAC * h
              and (y <= xeh or y + ch >= h - xeh))
        if v or hsh or xv or xh:
            out[lbl == i] = 255
    return out


# NOTE: a column/row-mean "bright, dark, recovers" profile rule was tried
# here to catch shadows that touch (and connected-component-merge with) real
# content on dense photo/diagram pages -- e.g. img20260427_0010, where the
# shadow merges into the page's dominant ink blob and the component rule
# above can't isolate it. It was reverted: real corpus content (sampled
# across all 5 books, ~135 of ~150 pages) produces the same bright/dark/
# bright column-mean shape often enough -- text blocks, photo edges, rules --
# that a width-only discriminator still misclassified real content as shadow
# on a book reported as having NO shadows (img20260525_0005). Don't re-add
# this without a discriminator validated against that negative case and a
# broad corpus sweep (diff-pixel count vs the unmodified image) showing no
# large-area false positives; see this commit's history for the failed
# attempt and the false-positive evidence.


# --- Cross-page (document-level) shadow-band detection -----------------------
# The per-page rules above cannot tell, on a single page, whether a dark column
# hugging the edge is a binding/ADF shadow or real content -- which is why the
# 2026-06-21 single-page attempt false-fired on dense text. The discriminating
# signal is CROSS-PAGE CONSISTENCY: an ADF shadow sits at the SAME absolute
# position on every sheet (fixed feed position), so it shows up as a dark column
# at a near-constant x on a large fraction of the book's pages, whereas a body/
# margin text column wanders page to page. Measured on tmp0613: the left shadow
# of img20260427_0010 is a dark column at x=31 with std 0.0 across all sampled
# pages. detect_shadow_bands() aggregates per-column dark frequency over the
# whole document and keeps only thin, outer-edge columns that recur on enough
# pages; apply_shadow_bands() whitens them. Positions are stored as fractions of
# width/height so the model transfers across the work-raster / original scales.
SHADOW_BAND_EDGE_FRAC = 0.07     # only consider the outer 7% of each side
SHADOW_BAND_MIN_H = 0.10         # a "dark column" has ink over >=10% of its height
SHADOW_BAND_PAGE_FRAC = 0.50     # ... and must recur on >=50% of pages to be a shadow
SHADOW_BAND_MAX_W = 0.05         # safety: a shadow band is < 5% of the page wide
SHADOW_BAND_MIN_PAGES = 8        # need at least this many pages to trust frequency
SHADOW_BAND_HALO_FRAC = 0.004    # whiten this much beyond the band (penumbra)


def _dark_columns(gray: np.ndarray, axis: int) -> np.ndarray:
    """Boolean profile over `axis` (0=columns/x, 1=rows/y): True where the line
    has ink (<= ink_threshold) spanning >= SHADOW_BAND_MIN_H of the cross extent,
    restricted to the outer SHADOW_BAND_EDGE_FRAC at either end."""
    binv = (gray <= ink_threshold(gray)).astype(np.uint8)
    cross = binv.shape[0] if axis == 0 else binv.shape[1]
    counts = binv.sum(axis=0 if axis == 0 else 1)
    n = counts.size
    dark = counts >= SHADOW_BAND_MIN_H * cross
    ew = int(n * SHADOW_BAND_EDGE_FRAC)
    mask = np.zeros(n, dtype=bool)
    if ew > 0:
        mask[:ew] = dark[:ew]
        mask[n - ew:] = dark[n - ew:]
    return mask


def detect_shadow_bands(grays) -> "list[tuple]":
    """Find binding/ADF shadow bands that recur at a consistent position across
    the document. `grays` is any iterable of single-channel page images (a
    generator is fine -- iterated once, one image held at a time, so a whole
    book's pages need not be materialized). Returns a list of
    (axis, lo_frac, hi_frac); axis 0 = a vertical band at x in [lo,hi]*W.
    Empty when no consistent thin edge band exists (the common case).

    VERTICAL bands only. ADF/binding shadows are left/right, and every shadow
    reported on the real corpus is vertical. A horizontal cross-page band
    almost always catches a RUNNING HEAD / footer instead -- e.g.
    img20260525_0005's "059  第3章 ..." chapter line recurs at a constant y and
    spans the text width, so horizontal detection would whiten it (and the
    nombre). Restricting to vertical removes that whole false-positive class
    while still catching every reported shadow.
    """
    BINS = 1000
    freq = np.zeros(BINS, dtype=np.float64)
    npages = 0
    for g in grays:
        if g is None or g.ndim != 2 or not g.size:
            continue
        mask = _dark_columns(g, 0)
        if mask.size != BINS:
            xs = np.linspace(0, 1, mask.size)
            binned = np.interp(np.linspace(0, 1, BINS), xs, mask.astype(float)) >= 0.5
        else:
            binned = mask
        freq += binned
        npages += 1
    if npages < SHADOW_BAND_MIN_PAGES:
        return []
    freq /= npages
    shadow_bins = freq >= SHADOW_BAND_PAGE_FRAC
    bands: "list[tuple]" = []
    i = 0
    while i < BINS:
        if not shadow_bins[i]:
            i += 1
            continue
        j = i
        while j < BINS and shadow_bins[j]:
            j += 1
        lo_frac, hi_frac = i / BINS, j / BINS
        if (hi_frac - lo_frac) <= SHADOW_BAND_MAX_W:
            bands.append((0, lo_frac, hi_frac))
        i = j
    return bands


def apply_shadow_bands(img: np.ndarray, bands: "list[tuple]") -> np.ndarray:
    """Whiten the document-level shadow bands (with a small halo) on one image."""
    if not bands:
        return img
    out = img.copy()
    h, w = img.shape[:2]
    for axis, lo, hi in bands:
        if axis == 0:
            halo = max(2, int(w * SHADOW_BAND_HALO_FRAC))
            a = max(0, int(lo * w) - halo); b = min(w, int(hi * w) + halo)
            out[:, a:b] = 255
        else:
            halo = max(2, int(h * SHADOW_BAND_HALO_FRAC))
            a = max(0, int(lo * h) - halo); b = min(h, int(hi * h) + halo)
            out[a:b, :] = 255
    return out


# --- Region-based (per-page) shadow removal -----------------------------------
# Replaces the position-magic "outer 7%" band: the boundary of where a shadow
# can be is derived from the page's own OCR text region, not a fixed fraction.
# A binding/ADF shadow is a thin, elongated dark STROKE in the LEFT/RIGHT margin
# -- i.e. OUTSIDE the text bounding box. Classified by SHAPE (tall and thin: a
# line segment), not by an absolute run-length: a long-run-length gate misses a
# shadow that's short, tapering, or fragmented into dashes by scan dust/a worn
# ADF roller (each dash individually still a thin elongated stroke, just not
# long enough on its own, and too far from its neighbors to bridge a gap into
# one run). Sparse marginal text -- isolated, roughly dot-shaped glyphs -- has a
# low height/width ratio and is kept. Detected figure/photo regions are
# SUBTRACTED from the result so a figure that bleeds to the edge is never
# whitened, while a shadow in the margin beside it still is. Validated on
# tmp0613 (all 5 books): catches every reported shadow page, zero pixels
# whitened inside any text or figure/photo region. No cross-page pass, no
# fixed-position constant.
SHADOW_MIN_HEIGHT_PX = 15    # below this height, a mark is noise/a text dot
SHADOW_ASPECT_MIN = 6        # height / width >= this to count as a "line"
# A binding/ADF shadow line is rarely perfectly straight -- it wobbles a few px
# side to side along its length (scan skew, a slightly bent original). A small
# halo leaves the wobble's far excursions as their own short, separate "line"-
# shaped slivers just outside the cleaned band (same physical line, missed).
SHADOW_COL_HALO_PX = 8       # whiten this many px around the detected line


def _text_bbox(regions):
    xs0, ys0, xs1, ys1 = [], [], [], []
    for r in regions or []:
        if r.kind != RegionKind.TEXT or r.box is None:
            continue
        xs0.append(r.box.x0); ys0.append(r.box.y0)
        xs1.append(r.box.x1); ys1.append(r.box.y1)
    if not xs0:
        return None
    return Box(min(xs0), min(ys0), max(xs1), max(ys1))


def region_shadow_mask(gray: np.ndarray, regions) -> np.ndarray:
    """Boolean mask of binding/ADF shadow pixels in the left/right margins
    (outside the OCR text box, excluding figure/photo regions)."""
    h, w = gray.shape
    tb = _text_bbox(regions)
    out = np.zeros((h, w), dtype=bool)
    # No TEXT region at all (a blank/near-blank leaf, or a page OCR found
    # nothing on) means there's no text box to scan OUTSIDE of -- but it also
    # means there's no real text to accidentally erase, so the whole page is
    # fair game. Returning empty here (the old behavior) left exactly these
    # pages' binding/ADF shadow lines undetected, which then leaked into
    # find_content_box's ink bbox (nothing else constrains it) and inflated
    # that page's crop -- the "page sizes aren't uniform" symptom.
    bands = ([(0, w)] if tb is None else
             ((0, max(0, int(tb.x0))), (min(w, int(tb.x1)), w)))
    dark = (gray <= ink_threshold(gray)).astype(np.uint8)
    halo = SHADOW_COL_HALO_PX
    # Bridge tiny (a few px) vertical gaps before labeling components: a scan-
    # dust speck or anti-aliasing dip can break one physical stroke into 2-3
    # fragments a couple pixels apart. DILATE only (not a full close): closing's
    # erode half can eat into the genuine top/bottom tip of a long run (an
    # asymmetric-kernel artifact). Dilate is monotonic -- it only ever adds
    # coverage, never erodes a real pixel. The gap bridged here (10px) is far
    # smaller than the spacing between genuinely separate marginal marks (e.g.
    # sparse text dots, tens of px apart), so it never merges unrelated content.
    gap_kernel = np.ones((10, 1), np.uint8)
    for a, b in bands:
        if b - a < 1:
            continue
        # Include a strip on the content side while labeling. OCR boxes can end
        # a few pixels inside a glyph; labeling only the nominal margin severs
        # that outside sliver from the rest of the glyph and makes the sliver
        # look exactly like a thin vertical shadow. A real shadow remains a
        # disconnected component, whereas real ink crossing the OCR boundary
        # is connected back into this guard strip and must be protected.
        guard = max(32, 2 * halo + 2)
        ea = max(0, a - guard) if a > 0 else a
        eb = min(w, b + guard) if b < w else b
        band = cv2.dilate(dark[:, ea:eb], gap_kernel)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
        protected_labels: set[int] = set()
        if a > ea:  # right margin: protect components reaching left into content
            protected_labels.update(int(v) for v in np.unique(lbl[:, :a - ea]) if v)
        if eb > b:  # left margin: protect components reaching right into content
            protected_labels.update(int(v) for v in np.unique(lbl[:, b - ea:]) if v)
        line_labels = [
            i for i in range(1, n)
            if stats[i, 3] >= SHADOW_MIN_HEIGHT_PX
            and stats[i, 3] >= SHADOW_ASPECT_MIN * stats[i, 2]
            and i not in protected_labels
        ]
        if not line_labels:
            continue
        comp = np.isin(lbl, line_labels)
        comp = cv2.dilate(comp.astype(np.uint8), np.ones((1, 2 * halo + 1), np.uint8)) > 0
        out[:, a:b] |= comp[:, a - ea:b - ea]
    # protect figures/photos: never whiten inside a non-text region
    for r in regions or []:
        if r.kind == RegionKind.TEXT or r.box is None:
            continue
        bx0 = max(0, int(r.box.x0)); by0 = max(0, int(r.box.y0))
        bx1 = min(w, int(r.box.x1)); by1 = min(h, int(r.box.y1))
        out[by0:by1, bx0:bx1] = False
    return out


def remove_region_shadows(img: np.ndarray, regions) -> np.ndarray:
    """Whiten region-based margin shadows on a gray or BGR image."""
    if not regions:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = region_shadow_mask(gray, regions)
    if not mask.any():
        return img
    out = img.copy()
    out[mask] = 255
    return out


# A near-blank page (mostly empty leaf, part-title, show-through) can keep a thin
# binding/ADF line along a side that the single-component shadow rule above misses
# because the line is *fragmented* (broken into dashes with gaps too large for any
# morphological kernel). Detect it by column projection: a column whose ink spans
# >=25% of the height top-to-bottom (gaps notwithstanding) is a line column.
# The discriminator from real margin content is POSITION: a binding line hugs the
# extreme paper edge (outer ~5%), whereas margin text columns, running-head tabs
# and nombre sit further in. So detection is confined to the outer band, and short
# items (tabs/nombre, <25% tall) are excluded by the extent test, leaving only the
# edge line -- which is whitened (whole column + halo). Gated on near-blank so dense
# pages are never altered. (Validated on tmp0613: removes the binding lines on the
# near-blank pages, leaves the 縦書き text column on 0005 p7 intact.)
#
# REVERTED 2026-06-21: tried removing the density gate to catch the same
# fragmented pattern on dense pages too (img20260427_0001 p74/90/92/118) --
# a 5-book diff-pixel sweep showed catastrophic over-triggering (up to
# ~550,000 changed px/page on img20260427_0010, vs a true positive's few
# hundred), so the gate must stay. On a dense page, a column's "ink spans
# >=25% of height with gaps allowed" condition is satisfied constantly by
# ordinary running text (many short ink runs spread down a column of body
# text), not just by a binding line -- the density gate is what makes that
# condition mean "binding line" instead of "any column with text in it".
# Don't remove this gate without a per-column discriminator that works on
# dense pages specifically (validated against a broad corpus sweep first).
EDGE_LINE_TRIGGER_FRAC = 0.015   # only on pages with <=1.5% ink (near-blank)
EDGE_LINE_BAND_FRAC = 0.07       # detect only in the outer 7% (extreme edge), so
#                                  inner margin content is separated by position
#                                  (7% catches a skewed line's inward-drifting end
#                                  while still excluding margin text at >=10% in)
EDGE_LINE_MIN_H = 0.10           # a line column's ink spans >=10% of the page height
#                                  (low is safe: only the binding line lives this far
#                                  out -- catches a steep line's inward-drifting end)
EDGE_LINE_MIN_COL_INK = 5        # a candidate column has at least this many ink px
EDGE_LINE_MAX_GROUP = 0.05       # safety cap: a line group is < 5% of page width
EDGE_LINE_HALO_FRAC = 0.002      # whiten this many px BEYOND the group (penumbra)


def _remove_fragmented_edge_lines(img: np.ndarray) -> np.ndarray:
    """Whiten a thin binding/ADF line in the L/R margin of a near-blank page.

    Detection is by column projection (a column whose ink spans >=25% of the page
    height is a line column), and a grouped run of line columns is whitened (whole
    column + halo) only when its average horizontal ink thickness is small -- a
    slant-tolerant test that keeps a thin binding line while sparing a (much
    thicker) 縦書き text column. Gated on near-blank so dense pages are untouched.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    thr = ink_threshold(gray)
    binv = (gray <= thr).astype(np.uint8)
    if binv.mean() > EDGE_LINE_TRIGGER_FRAC:        # not near-blank -> leave it
        return img
    h, w = gray.shape
    ew = int(w * EDGE_LINE_BAND_FRAC)
    halo = max(2, int(w * EDGE_LINE_HALO_FRAC))
    out = img.copy()
    cols = list(range(0, ew)) + list(range(w - ew, w))
    line_cols = []
    for x in cols:
        ys = np.flatnonzero(binv[:, x])
        if ys.size >= EDGE_LINE_MIN_COL_INK and (ys[-1] - ys[0]) >= EDGE_LINE_MIN_H * h:
            line_cols.append(x)
    if not line_cols:
        return out
    # group consecutive line columns (bridge gaps up to the halo), whiten thin ones
    groups = []
    a = p = line_cols[0]
    for x in line_cols[1:]:
        if x - p <= halo:
            p = x
        else:
            groups.append((a, p)); a = p = x
    groups.append((a, p))
    for a, b in groups:
        if (b - a + 1) <= EDGE_LINE_MAX_GROUP * w:  # thin edge group => binding line
            out[:, max(0, a - halo):min(w, b + 1 + halo)] = 255
    return out


def _deskewed_binary(img_bgr: np.ndarray, deskew: Deskew) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    gray = remove_edge_shadows(gray)         # drop shadows before measuring content
    if abs(deskew.angle_deg) > 1e-3:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deskew.angle_deg, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderValue=255)
    # A raw per-page Otsu degenerates on a near-blank page: with almost no real
    # ink, Otsu finds its split point in the high-200s (splitting noise from
    # background, not ink from background), so faint anti-aliasing/scan noise
    # gets classified as content. ink_threshold's standalone fallback guards
    # exactly against this (falls back to a fixed safe floor when Otsu sits
    # high AND genuinely dark pixels are vanishingly rare).
    th = ink_threshold(gray)
    binary = np.where(gray <= th, 255, 0).astype(np.uint8)
    return binary


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


MARGINAL_SPAN_GAP_FRAC = 0.02   # text spans this far apart (frac of content
#                                 width) are separate blocks, not one block
MARGINAL_SPAN_FRAC = 0.25       # a span narrower than this (frac of content
#                                 width) sitting apart from a dominant body
#                                 block is furniture (running head/柱, nombre)
DOMINANT_SPAN_FRAC = 0.50       # ... but only strip furniture when ONE span is
#                                 at least this wide (a real horizontal body
#                                 block); tategaki column pages have no such
#                                 dominant span and keep their whole content box


def body_text_box(params) -> "Box | None":
    """The main text block, EXCLUDING marginal furniture (a running head / 柱
    hugging the fore-edge, the page number) that sits apart from the body in
    the margin. Used for centering + format sizing so a fore-edge 柱 doesn't
    drag the whole block off-centre (real-corpus issue: img20260430_0002).

    Found from the horizontal spans of the text regions: merge regions into
    x-spans; if ONE span dominates the content width (a horizontal body block,
    yokogaki), keep the wide spans and drop the narrow separated ones (柱 /
    nombre); otherwise (no dominant span -- e.g. tategaki columns, or a
    figure page) fall back to the full content box, so nothing is wrongly
    stripped. The body's y-extent comes from the kept regions too."""
    content = params.margin.content if params.margin else None
    if content is None:
        return None
    regions = [r for r in (params.regions or []) if r.kind == RegionKind.TEXT]
    if not regions or content.width <= 0:
        return content
    ivs = sorted((r.box.x0, r.box.x1) for r in regions)
    gap = MARGINAL_SPAN_GAP_FRAC * content.width
    merged: list[list[float]] = []
    for a, b in ivs:
        if merged and a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if max(b - a for a, b in merged) < DOMINANT_SPAN_FRAC * content.width:
        return content                       # no dominant block -> keep all
    keep = [(a, b) for a, b in merged
            if (b - a) >= MARGINAL_SPAN_FRAC * content.width]
    if not keep:
        return content
    x0 = min(a for a, _ in keep)
    x1 = max(b for _, b in keep)
    ys = [(r.box.y0, r.box.y1) for r in regions
          if any(a - 1 <= (r.box.x0 + r.box.x1) / 2 <= b + 1 for a, b in keep)]
    if not ys:
        return content
    return Box(x0, min(y for y, _ in ys), x1, max(y for _, y in ys))


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
    - the BASELINE for every page is centered on both axes: this page's own
      content centered within the document-uniform crop. Averaged over the
      whole book this already centers the body on the page, since it's each
      page's actual content extent against the one shared crop size.
    - when a page's nombre is CONFIDENT (a page number was assigned by the OCR
      sequence resolver), a small CORRECTION is added on top of that
      baseline so the nombre lands at a common per-parity offset from the
      baseline -- removing the residual per-page jitter from a varying
      content box, without dragging the whole page off-center. (Anchoring
      directly to "nombre offset from content's near edge" instead of to the
      centered baseline was tried first and is wrong: the "slack" between
      this page's content size and the document-wide uniform crop size then
      lands entirely on the far side, biasing every page toward one corner
      -- exactly what real-corpus review caught.)
    Either way the offset is clamped so every side keeps >= the output margin.
    """
    have = [p for p in pages if p.margin and p.margin.content]
    if not have:
        return
    use_dpi = all(p.dpi for p in have)

    def extent(p, px):  # content size in inches (if dpi) else pixels
        return px / p.dpi if use_dpi else px

    def confident(p):
        # nombre_box alone is enough to anchor on: nombre.resolve sets it
        # either when the digit VALUE is trusted (page_number is also set) or,
        # for a page whose own numbering run is too short to be trusted by
        # value (e.g. front matter), when the box still sits in the position
        # model's expected slot -- geometry-only, page_number stays None. Both
        # cases are a real nombre worth anchoring margins on.
        return p.margin.nombre_box is not None

    # ONE size for the whole document -- non-negotiable (page-format
    # uniformity is the point; a real fold-out is the only legitimate
    # exception, and even that should be a flagged human decision, not a
    # silently-bigger page). Two failure modes to avoid when judging it:
    #  (a) detection FAILURE (confidence 0.0) falls back to the full raw page
    #      as "content", which is not a real body-text extent -- exclude those.
    #  (b) the raw MAX is pulled up by a single legitimate-but-large outlier
    #      page (a fold-out, a dense full-bleed chapter-divider graphic, a
    #      detection that over-grew), which would inflate the OUTPUT SIZE for
    #      the whole document if naively used as-is. Measured on tmp0613:
    #      img20260430_0002 has a circular full-bleed chapter-divider graphic
    #      whose content height (2469px) is genuinely ~175px taller than every
    #      other page's P97.5 (2294px).
    # So the uniform TARGET size is the P97.5 of reliable content extents,
    # robust to a few extreme pages without the instability of mean+2sigma
    # (sigma is itself inflated by the very outliers being discounted; on a
    # tight book mean+2sigma can even exceed the max). The <=2.5% of pages
    # whose own content exceeds this are NOT clipped and NOT given a bigger
    # page: crop_size() below still floors their SOURCE region at their own
    # content (so nothing is lost), but compose_transform (render.py) then
    # SHRINKS that larger source region to fit the same uniform target_w/
    # target_h as everyone else -- see Margin.target_w/target_h.
    reliable = [p for p in have if p.margin.confidence >= 0.2] or have
    UNIFORM_PCT = 97.5
    uni_w = float(np.percentile([extent(p, p.margin.content.width) for p in reliable], UNIFORM_PCT))
    uni_h = float(np.percentile([extent(p, p.margin.content.height) for p in reliable], UNIFORM_PCT))

    # Per-page body box (furniture-stripped, see body_text_box) -- the thing we
    # actually centre and size the format from, so a fore-edge 柱 can't drag
    # the block off-centre. Cached per page id.
    body_boxes = {id(p): body_text_box(p) for p in have}

    def bbox(p):
        return body_boxes.get(id(p)) or p.margin.content

    # Robust layout statistics, measured ONLY from numbered body pages (a
    # cover/divider's box is deliberately off-position and would pollute them;
    # those pages instead keep the proportional baseline() path below).
    #
    #   M  -- ONE symmetric side margin: the average of the clean left & right
    #         BODY margins (the 柱 is already excluded from the body box, so
    #         BOTH sides are clean here). Centering the body in body_w + 2M
    #         gives equal left/right margins -- the whole point, since perfect
    #         deskew makes any L/R imbalance jump out.
    #   a_top / a_bottom -- independent head/foot margins (this asymmetry, e.g.
    #         running-header space, is a real design choice, kept).
    #   d  -- how far the nombre's OUTER edge sticks out past the body's
    #         fore-edge; lets us anchor on the crisp nombre while still landing
    #         the body's fore margin at M.
    default_margin_in = output_margin_mm / 25.4
    stat_pages = [p for p in reliable if confident(p) and p.margin.page_w and p.margin.page_h]
    body_w_vals, body_h_vals, side_margin_vals = [], [], []
    top_vals, bottom_vals, d_vals = [], [], []
    by_parity_nombre_x: dict[int, list[float]] = {0: [], 1: []}
    nombre_y_fracs: list[float] = []
    for p in stat_pages:
        bb = bbox(p)
        pw, ph = p.margin.page_w, p.margin.page_h
        nb = p.margin.nombre_box
        body_w_vals.append(extent(p, bb.width))
        body_h_vals.append(extent(p, bb.height))
        side_margin_vals.append(extent(p, bb.x0))           # left body margin
        side_margin_vals.append(extent(p, pw - bb.x1))      # right body margin
        top_vals.append(extent(p, bb.y0))
        bottom_vals.append(extent(p, ph - bb.y1))
        by_parity_nombre_x[p.source.page_index % 2].append((nb.x0 + nb.x1) / 2 / pw)
        # nombre outer edge vs body fore edge (sign: outward past the body)
        if (nb.x0 + nb.x1) / 2 > (bb.x0 + bb.x1) / 2:        # nombre on the right
            d_vals.append(extent(p, nb.x1 - bb.x1))
        else:                                               # nombre on the left
            d_vals.append(extent(p, bb.x0 - nb.x0))
        nombre_y_fracs.append(((nb.y0 + nb.y1) / 2 - bb.y0) / (bb.height or 1.0))

    body_w = float(np.median(body_w_vals)) if body_w_vals else uni_w
    body_h = float(np.median(body_h_vals)) if body_h_vals else uni_h
    side_margin = float(np.median(side_margin_vals)) if side_margin_vals else default_margin_in
    top_margin = float(np.median(top_vals)) if top_vals else default_margin_in
    bottom_margin = float(np.median(bottom_vals)) if bottom_vals else default_margin_in
    d_med = float(np.median(d_vals)) if d_vals else 0.0
    # the nombre's own (visible) fore margin that keeps the body's fore margin
    # at the symmetric side_margin: side_margin = nombre_fore_margin + d
    nombre_fore_margin = side_margin - d_med

    # Which side carries the nombre, per parity, IS the fore-edge (a page
    # number is conventionally set at the fore-edge corner). A direct, reliable
    # signal -- unlike comparing raw L/R gap magnitudes, which picks the wrong
    # side on any page whose content detection is a little noisy on one edge.
    nombre_side: dict[int, str] = {}
    for parity, xs in by_parity_nombre_x.items():
        if xs:
            nombre_side[parity] = "right" if float(np.median(xs)) > 0.5 else "left"
    nombre_near_bottom = (
        float(np.median(nombre_y_fracs)) > 0.5 if nombre_y_fracs else True)

    def target_size(p):  # the FIXED uniform output size, in this page's own pixels
        dpi = p.dpi if use_dpi else 1.0
        pad_w, pad_h = 2 * side_margin, top_margin + bottom_margin
        return (body_w + pad_w) * (dpi if use_dpi else 1), (body_h + pad_h) * (dpi if use_dpi else 1)

    def crop_size(p):
        # SOURCE region size = the uniform target, but never narrower than this
        # page's own FULL content box (柱 included), so the crop always
        # CONTAINS the 柱 even though only the body box was centred -- the 柱
        # rides along inside the side margin (it protrudes less than
        # side_margin past the body, so it fits without widening the format).
        # Only a genuine outlier whose whole content exceeds the uniform target
        # makes crop > target; compose_transform then shrinks it to target,
        # never clips. target_size already adds the margins, so do NOT add them
        # again to content.width here -- doing so silently inflated the format
        # by the 柱's width and broke the centering.
        tw, th = target_size(p)
        if p.margin.confidence >= 0.2:
            c = p.margin.content
            fl = output_margin_mm / 25.4 * (p.dpi if use_dpi else 96)
            return max(tw, c.width + 2 * fl), max(th, c.height + 2 * fl)
        return tw, th

    def baseline(p):
        # ⑤/⑥ fallback for a page with no usable nombre (cover, divider, OCR
        # miss): keep its content at the SAME proportional position it had on
        # the original page (a deliberately off-centre part-title stays where
        # it was); with no stored page size, dead-centre instead.
        c = p.margin.content
        cw, ch = crop_size(p)
        cx, cy = c.x0 + c.width / 2, c.y0 + c.height / 2
        if p.margin.page_w and p.margin.page_h:
            fx, fy = cx / p.margin.page_w, cy / p.margin.page_h
        else:
            fx, fy = 0.5, 0.5
        return cx - fx * cw, cy - fy * ch

    for p in have:
        dpi = p.dpi if use_dpi else 1.0
        unit = dpi if use_dpi else 1
        margin_px = output_margin_mm / 25.4 * (p.dpi if use_dpi else 96)
        crop_w, crop_h = crop_size(p)
        c = p.margin.content
        bb = bbox(p)
        parity = p.source.page_index % 2
        side = nombre_side.get(parity)
        # HORIZONTAL: anchor on the nombre's outer edge so it lands at the same
        # visible fore margin (nombre_fore_margin) on every same-parity page --
        # the user-chosen priority. Because that margin is derived to put the
        # body's fore edge at the symmetric side_margin, the body is centred
        # for the median page; per-page body-width variation lands on the
        # GUTTER (binding) side, the least visible edge.
        if confident(p) and side is not None:
            nb = p.margin.nombre_box
            if side == "left":
                x0 = nb.x0 - nombre_fore_margin * unit
            else:
                x0 = nb.x1 + nombre_fore_margin * unit - crop_w
        else:
            x0 = baseline(p)[0]
        # VERTICAL: anchor on the nombre's bottom edge (y1) -- digits share a
        # baseline, not a cap-height, so this keeps the printed numeral level.
        if confident(p):
            nb = p.margin.nombre_box
            if nombre_near_bottom:
                y0 = nb.y1 + bottom_margin * unit - crop_h
            else:
                y0 = nb.y1 - top_margin * unit
        else:
            y0 = baseline(p)[1]
        # Never clip real content: if the nombre anchor would push the FULL
        # content box (柱 included) outside the crop, shift the crop minimally
        # to contain it -- the user's "判型外に出る要素があれば中心をずらす".
        # A confident page floors at 0 (only prevent clipping, don't fight the
        # shared anchor for cosmetic headroom); a baseline page keeps the small
        # output-margin floor. Skipped for a detection-failed page (its content
        # is the synthetic full page, wider than the uniform crop).
        # Floor: a nombre-anchored page only needs "don't clip" (floor 0) -- it
        # already sits at the large robust side/top/bottom margin, and forcing
        # the small output-margin floor on top could only fight that shared
        # anchor. A baseline (no-nombre) page keeps the real output-margin
        # floor, since there's no anchor to protect instead.
        if p.margin.confidence >= 0.2:
            xf = 0.0 if (confident(p) and side is not None) else margin_px
            yf = 0.0 if confident(p) else margin_px
            x0 = max(min(x0, c.x0 - xf), c.x1 + xf - crop_w)
            y0 = max(min(y0, c.y0 - yf), c.y1 + yf - crop_h)
        p.margin.crop = Box(x0, y0, x0 + crop_w, y0 + crop_h)
        target_w, target_h = target_size(p)
        p.margin.target_w, p.margin.target_h = target_w, target_h
        # crop bigger than target = this page's own content didn't fit the
        # uniform size; compose_transform shrinks it down -- flag for review.
        if (crop_w > target_w * 1.001 or crop_h > target_h * 1.001) \
                and Flag.CONTENT_SCALED_DOWN not in p.flags:
            p.flags.append(Flag.CONTENT_SCALED_DOWN)


MARGIN_OUTLIER_RATIO = 1.30   # content >30% larger than the median = an outlier


def align_margins(margins: list[Margin]) -> list[Flag | None]:
    """Cross-page consistency check. Returns a per-page flag (or None).

    Uniform-size rendering + the >= margin clamp + margin fill make the output
    robust to ordinary content-box variation (a photo page is simply bigger), so
    we no longer flag every small departure -- that produced a flood of false
    reviews. The one residual risk is an OUTLIER-LARGE content box (a bad
    detection that bloats the whole document's uniform crop), so we flag only
    pages whose content is much larger than the median. (Tiny / failed boxes are
    caught earlier as MARGIN_NOT_FOUND via confidence.)
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
        if (m.content.width > MARGIN_OUTLIER_RATIO * med_w
                or m.content.height > MARGIN_OUTLIER_RATIO * med_h):
            flags.append(Flag.MARGIN_INCONSISTENT)
        else:
            flags.append(None)
    return flags
