"""Content box and margin normalization (margin v2, 2026-11-22).

Both produce parameters in ORIGINAL-image pixels:

1. content_ink_box(): every printed mark on the page, ScanTailor-style --
   all ink counts as content unless it is recognisably not print (scan
   surround touching the border, thin edge lines, dust in the outer band,
   isolated dots), plus the OCR/layout regions. The renderer whitens what
   lies outside, so this box must be inclusive.

2. normalize_margins(): one page size for the book, with the type area
   (版面) registered at one position. The anchor is the corner where the text
   starts (top-right in tategaki, top-left in yokogaki), taken from the long
   OCR text lines and accepted only where it agrees with the book's
   consensus; other pages keep their proportional position on the paper.

Shadow handling (remove_edge_shadows, region and cross-page bands) runs
before both. Earlier versions built the content box from OCR regions only
and anchored pages on the page number; both failed on real books (print
outside OCR regions whitened, misread page numbers pushing pages against the
edge) -- see docs/OCR_CONSOLIDATION_PLAN.md.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..model import Box, Deskew, Flag, Margin, RegionKind

SPECKLE_AREA_FRAC = 1e-5     # components smaller than this fraction are noise
NOMBRE_BAND_FRAC = 0.08      # top/bottom 8% of the page is the nombre band
NOMBRE_MAX_HEIGHT_FRAC = 0.04


# A shadow STARTS at the paper edge: measured on 9 books (tmp0613 + img20261121)
# every real one begins within 1.3% of the edge, while long thin marks that
# begin 4-14% in are print -- a heading band, the rule under a running head,
# the top border of a table, a band of white-on-black text -- and the old 15%
# window whitened them (img20260423_0001 p21/p127/p253, img20261121_0002 p268,
# 0013 p9).
SHADOW_EDGE_FRAC = 0.03
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


LINE_ASPECT_MAX = 15.0        # longer/shorter side above this = a rule/edge line
LINE_THIN_FRAC = 0.004        # ... when also thinner than this (x page side)


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


# Garbage rules for content_ink_box (ScanTailor-style: everything printed is
# content unless it is recognisably NOT print). Fractions of the page side.
BLEED_INWARD_FRAC = 0.08      # a border-touching blob reaching this far in is
BLEED_MIN_FRAC = 0.02         #   a bleed picture (if also this thick), not scan surround
EDGE_ZONE_FRAC = 0.03         # outer band where specks and rule fragments are dust
EDGE_LINE_ZONE_FRAC = 0.06    # outer band where a long thin line is a scan edge line
TINY_AREA_FRAC = 3e-5         # a component this small is a dot ...
ISOLATED_FRAC = 0.02          # ... and garbage when nothing else is within this


def content_ink_box(img_bgr: np.ndarray, deskew: Deskew, regions=()) -> "Box | None":
    """Bounding box (deskewed frame) of every printed mark on the page.

    Inclusive by design, like ScanTailor's content box: all ink counts unless
    it is recognisably not print -- scanner surround touching the image border,
    thin edge lines, dust in the outer band, isolated dots. The OCR/layout
    regions are added on top. What lies outside this box is whitened by the
    renderer, so a page number on a black tab, a rule, an ornament or a
    running head that OCR did not report as text is kept (the old region-only
    box whitened them: img20261121_0012 p7)."""
    binary = _deskewed_binary(img_bgr, deskew)
    h, w = binary.shape
    side = float(min(h, w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    speck = SPECKLE_AREA_FRAC * h * w
    tiny = TINY_AREA_FRAC * h * w
    ez_x, ez_y = EDGE_ZONE_FRAC * w, EDGE_ZONE_FRAC * h
    lz_x, lz_y = EDGE_LINE_ZONE_FRAC * w, EDGE_LINE_ZONE_FRAC * h
    kept, dots = [], []
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        x1, y1 = x + cw, y + ch
        if area < speck:
            continue
        if x == 0 or y == 0 or x1 == w or y1 == h:
            # scan surround / facing-page text, unless a bleed picture that
            # reaches well into the page
            reach = max(x1 if x == 0 else 0, w - x if x1 == w else 0,
                        y1 if y == 0 else 0, h - y if y1 == h else 0)
            limit = BLEED_INWARD_FRAC * (w if (x == 0 or x1 == w) else h)
            if not (reach >= limit and min(cw, ch) >= BLEED_MIN_FRAC * side):
                continue
        in_edge = x1 <= ez_x or x >= w - ez_x or y1 <= ez_y or y >= h - ez_y
        if in_edge and area < 4 * tiny:
            continue                                  # dust in the outer band
        long_, short = max(cw, ch), max(1, min(cw, ch))
        in_line_zone = x1 <= lz_x or x >= w - lz_x or y1 <= lz_y or y >= h - lz_y
        if in_line_zone and long_ / short > LINE_ASPECT_MAX and short < LINE_THIN_FRAC * side:
            continue                                  # scan edge line
        (dots if area < tiny else kept).append((x, y, x1, y1))
    boxes = [Box(float(a), float(b), float(c), float(d)) for a, b, c, d in kept]
    # OCR regions count only clipped to the image and over printed ink: NDL
    # can report a text line on a blank page that runs off the paper
    # (img20261121_0011 p30, a "0" 1759 px wide on a 1415 px page), which
    # pushed the crop past the paper edge
    ink = np.zeros((h, w), np.uint8)
    for a, b, c, d in kept + dots:
        ink[b:d, a:c] = 1
    regions = [r for r in (regions or ()) if _region_on_ink(r.box, ink)]
    if dots and kept:
        # a dot counts only next to other print (punctuation, a diacritic,
        # the dot of a dotted rule); a lone one is dust
        mask = np.zeros((h, w), np.uint8)
        for a, b, c, d in kept:
            mask[b:d, a:c] = 1
        for r in regions or ():
            bx = r.box
            mask[max(0, int(bx.y0)):max(0, int(bx.y1)), max(0, int(bx.x0)):max(0, int(bx.x1))] = 1
        g = max(3, int(ISOLATED_FRAC * side))
        near = cv2.dilate(mask, np.ones((2 * g + 1, 2 * g + 1), np.uint8))
        for a, b, c, d in dots:
            if near[b:d, a:c].any():
                boxes.append(Box(float(a), float(b), float(c), float(d)))
    boxes += [_clip_box(r.box, w, h) for r in regions]
    if not boxes:
        return None
    return Box(min(b.x0 for b in boxes), min(b.y0 for b in boxes),
               max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def _clip_box(b: Box, w: int, h: int) -> Box:
    return Box(min(max(b.x0, 0.0), w), min(max(b.y0, 0.0), h),
               min(max(b.x1, 0.0), w), min(max(b.y1, 0.0), h))


def _region_on_ink(b: Box, ink: np.ndarray) -> bool:
    h, w = ink.shape
    c = _clip_box(b, w, h)
    x0, y0, x1, y1 = int(c.x0), int(c.y0), int(np.ceil(c.x1)), int(np.ceil(c.y1))
    return x1 > x0 and y1 > y0 and bool(ink[y0:y1, x0:x1].any())


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


LINE_ASPECT = 1.5          # a text region this elongated is a line (else a lone glyph)
BODY_LINE_FRAC = 0.35      # a body line is >= this x the book's long-line length (P90)
FULL_TYPE_FRAC = 0.8       # a page whose body spans >= this x the P90 is a full page
ANCHOR_TOL_IN = 0.2        # start corner within this (inch) of the consensus = aligned
ANCHOR_TOL_FRAC = 0.03     # ... same, as a page fraction when dpi is unknown
ANCHOR_MIN_GROUP = 3       # below this many pages a group has no consensus to test


def _line_dir(b: Box) -> "str | None":
    if b.height >= LINE_ASPECT * b.width:
        return "v"
    if b.width >= LINE_ASPECT * b.height:
        return "h"
    return None


def type_area_bodies(pages) -> dict:
    """id(page) -> (direction, body Box, line count) for pages with body text.

    The body is the box of the page's long text lines in its main direction
    ("v" tategaki columns, "h" yokogaki lines); running heads, page numbers,
    captions and headings are shorter and drop out. A book with no text
    regions at all (--no-ocr) uses each content box as its body."""
    texts = {id(p): [r.box for r in (p.regions or []) if r.kind == RegionKind.TEXT]
             for p in pages}
    if not any(texts.values()):
        return {id(p): ("h", p.margin.content, 0) for p in pages
                if p.margin and p.margin.content}
    lens: dict = {"v": [], "h": []}
    for boxes in texts.values():
        for b in boxes:
            d = _line_dir(b)
            if d:
                lens[d].append(max(b.width, b.height))
    ref = {d: float(np.percentile(v, 90)) for d, v in lens.items() if len(v) >= 5}
    out = {}
    for p in pages:
        lines: dict = {"v": [], "h": []}
        for b in texts[id(p)]:
            d = _line_dir(b)
            if d in ref and max(b.width, b.height) >= BODY_LINE_FRAC * ref[d]:
                lines[d].append(b)
        d = max(lines, key=lambda k: sum(max(b.width, b.height) for b in lines[k]))
        if lines[d]:
            bs = lines[d]
            out[id(p)] = (d, Box(min(b.x0 for b in bs), min(b.y0 for b in bs),
                                 max(b.x1 for b in bs), max(b.y1 for b in bs)), len(bs))
    return out


def _start_corner(d: str, b: Box) -> tuple:
    # where the text starts: top-right for tategaki, top-left for yokogaki.
    # Line heads and the first line are fixed by the type area (版面) even on
    # a short page, while line ends and the last line are not.
    return (b.x1, b.y0) if d == "v" else (b.x0, b.y0)


def normalize_margins(pages, output_margin_mm: float = 5.0) -> None:
    """Uniform page size with the type area (版面) at one position (sets margin.crop).

    A printed book sets the body text in the same type area on every page, and
    the running head, page number and notes at fixed places around it. So the
    pages are registered on the type area's start corner (the top-right of the
    first column in tategaki, the top-left of the first line in yokogaki):
    every page whose start corner agrees with the book's consensus position
    (per writing direction and recto/verso) puts that corner at the same
    output position, horizontally centred with equal left/right margins.
    Everything printed around the type area rides along. Earlier versions
    anchored on the page number, which a stray mark or a kanji of the running
    head could impersonate (img20261121_0010 p128, 0012 p7): the whole page
    was then pushed against the edge.

    Pages without an agreeing start corner (cover, part title, figure page,
    no text) keep the proportional position they had on the scanned paper.

    Size: the type area plus the book's median paper margins (equal sides),
    so the result looks like the printed page. A page whose content does not
    fit is never clipped: its crop grows and the renderer scales it down
    (flagged CONTENT_SCALED_DOWN)."""
    have = [p for p in pages if p.margin and p.margin.content]
    if not have:
        return
    use_dpi = all(p.dpi for p in have)

    def unit(p):  # pixels per output unit (inch, or pixel without dpi)
        return p.dpi if use_dpi else 1.0

    default_margin = output_margin_mm / 25.4 * (1.0 if use_dpi else 96.0)
    reliable = [p for p in have if p.margin.confidence >= 0.2] or have
    UNIFORM_PCT = 97.5
    uni_w = float(np.percentile([p.margin.content.width / unit(p) for p in reliable], UNIFORM_PCT))
    uni_h = float(np.percentile([p.margin.content.height / unit(p) for p in reliable], UNIFORM_PCT))

    bodies = type_area_bodies(have)
    sized = [p for p in reliable if id(p) in bodies and p.margin.page_w and p.margin.page_h]

    # -- which pages are registered: start corner near the group consensus --
    groups: dict = {}
    for p in sized:
        d, b, _ = bodies[id(p)]
        groups.setdefault((d, p.source.page_index % 2), []).append(p)
    anchored: dict = {}
    per_dir = {d: sum(len(ps) for k, ps in groups.items() if k[0] == d) for d in ("v", "h")}
    main_dir = max(per_dir, key=per_dir.get)
    for key, ps in groups.items():
        if len(ps) < ANCHOR_MIN_GROUP and key[0] != main_dir:
            # two yokogaki pages in a tategaki book (the cover, the colophon)
            # are no type area: registering them pushed the cover's title
            # into a corner (img20261121_0015 p1). Only a short book's own
            # direction is trusted without a consensus.
            continue
        pos = [(_start_corner(key[0], bodies[id(p)][1]), p) for p in ps]
        if len(ps) >= ANCHOR_MIN_GROUP:
            mx = float(np.median([c[0] / unit(p) for c, p in pos]))
            my = float(np.median([c[1] / unit(p) for c, p in pos]))
        for (cx, cy), p in pos:
            if len(ps) >= ANCHOR_MIN_GROUP:
                tol = ANCHOR_TOL_IN if use_dpi else ANCHOR_TOL_FRAC * max(p.margin.page_w, p.margin.page_h)
                if abs(cx / unit(p) - mx) > tol or abs(cy / unit(p) - my) > tol:
                    continue
            anchored[id(p)] = key[0]

    # -- type area size per direction, from full pages --
    def dims(d):
        ps = [p for p in sized if anchored.get(id(p)) == d]
        if not ps:
            return None
        ws = [bodies[id(p)][1].width / unit(p) for p in ps]
        hs = [bodies[id(p)][1].height / unit(p) for p in ps]
        fw = [w for w in ws if w >= FULL_TYPE_FRAC * float(np.percentile(ws, 90))]
        fh = [h for h in hs if h >= FULL_TYPE_FRAC * float(np.percentile(hs, 90))]
        full = [p for p, w, h in zip(ps, ws, hs)
                if w >= FULL_TYPE_FRAC * float(np.percentile(ws, 90))
                and h >= FULL_TYPE_FRAC * float(np.percentile(hs, 90))] or ps
        return float(np.median(fw)), float(np.median(fh)), ps, full

    tdims = {d: dims(d) for d in ("v", "h")}
    dom = max((d for d in tdims if tdims[d]), key=lambda d: len(tdims[d][2]), default=None)
    if dom is not None:
        tw, th, _, full = tdims[dom]
        side = float(np.median([(bodies[id(p)][1].x0 + p.margin.page_w - bodies[id(p)][1].x1)
                                / 2 / unit(p) for p in full]))
        top = float(np.median([bodies[id(p)][1].y0 / unit(p) for p in full]))
        bottom = float(np.median([(p.margin.page_h - bodies[id(p)][1].y1) / unit(p) for p in full]))
        side, top, bottom = (max(v, default_margin) for v in (side, top, bottom))
        W, H = tw + 2 * side, th + top + bottom
    else:
        W, H = uni_w + 2 * default_margin, uni_h + 2 * default_margin

    def anchor_out(d):  # output position of direction d's start corner
        tw_d, th_d = (tdims[d][0], tdims[d][1]) if tdims.get(d) else (W, H)
        left = (W - tw_d) / 2.0
        ax = left + tw_d if d == "v" else left
        if dom is not None and top + bottom > 0:
            ay = (H - th_d) * top / (top + bottom)
        else:
            ay = (H - th_d) / 2.0
        return ax, ay

    for p in have:
        u = unit(p)
        tw_px, th_px = W * u, H * u
        c = p.margin.content
        margin_px = output_margin_mm / 25.4 * (p.dpi if use_dpi else 96)
        # the crop only has to CONTAIN the content (print running close to
        # the paper edge -- a full-width rule, a cover -- is legitimate); it
        # grows, and the page is scaled down, only when the content itself
        # is larger than the page
        # (without paper statistics the page size is a guess, so keep the
        # output margin around the content as before)
        pad = 0.0 if dom is not None else 2 * margin_px
        if p.margin.confidence >= 0.2:
            crop_w, crop_h = max(tw_px, c.width + pad), max(th_px, c.height + pad)
        else:
            crop_w, crop_h = tw_px, th_px
        d = anchored.get(id(p))
        if d is not None:
            ax, ay = anchor_out(d)
            cx, cy = _start_corner(d, bodies[id(p)][1])
            x0, y0 = cx - ax * u, cy - ay * u
            floor = 0.0
        else:
            # proportional position on the paper (a deliberately off-centre
            # part title stays where it was); centred without a page size
            fx = (c.x0 + c.width / 2) / p.margin.page_w if p.margin.page_w else 0.5
            fy = (c.y0 + c.height / 2) / p.margin.page_h if p.margin.page_h else 0.5
            x0 = c.x0 + c.width / 2 - fx * crop_w
            y0 = c.y0 + c.height / 2 - fy * crop_h
            floor = margin_px
        if p.margin.confidence >= 0.2:
            # never clip: shift the crop minimally to contain the content,
            # keeping up to `floor` of white around it where there is room
            fx_ = min(floor, max(0.0, (crop_w - c.width) / 2.0))
            fy_ = min(floor, max(0.0, (crop_h - c.height) / 2.0))
            x0 = max(min(x0, c.x0 - fx_), c.x1 + fx_ - crop_w)
            y0 = max(min(y0, c.y0 - fy_), c.y1 + fy_ - crop_h)
        p.margin.crop = Box(x0, y0, x0 + crop_w, y0 + crop_h)
        p.margin.target_w, p.margin.target_h = tw_px, th_px
        # recomputed every time (remargin re-runs this on stored params)
        p.flags = [f for f in p.flags if f != Flag.CONTENT_SCALED_DOWN]
        if crop_w > tw_px * 1.001 or crop_h > th_px * 1.001:
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
