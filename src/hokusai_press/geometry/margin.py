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

from ..model import Box, Deskew, Flag, Margin

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


def ink_threshold(gray: np.ndarray) -> int:
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = int(otsu)
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
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

    def confident(p):   # the OCR resolver only assigns a number it trusts
        return p.page_number is not None and p.margin.nombre_box is not None

    # ONE size for the whole document (uniform judgment size). Two failure
    # modes to avoid:
    #  (a) detection FAILURE (confidence 0.0) falls back to the full raw page
    #      as "content", which is not a real body-text extent -- exclude those.
    #  (b) the raw MAX is pulled up by a single legitimate-but-large outlier
    #      page (a fold-out, a detection that over-grew), bloating the crop for
    #      the WHOLE document. Measured on tmp0613: e.g. img20260430_0002 width
    #      max=2295px vs P97.5~1700px (one outlier inflating all 160 pages).
    # So the uniform size is the P97.5 of reliable content extents -- robust to
    # a few extreme pages without the instability of mean+2sigma (sigma is
    # itself inflated by the very outliers we're discounting; on a tight book
    # mean+2sigma can even exceed the max). The <=2.5% of pages whose own
    # content exceeds the uniform size are NOT clipped: crop_size() floors each
    # page at its own content+margin (see below), so an outlier gets a slightly
    # larger page rather than losing content or enlarging every other page.
    reliable = [p for p in have if p.margin.confidence >= 0.2] or have
    UNIFORM_PCT = 97.5
    uni_w = float(np.percentile([extent(p, p.margin.content.width) for p in reliable], UNIFORM_PCT))
    uni_h = float(np.percentile([extent(p, p.margin.content.height) for p in reliable], UNIFORM_PCT))

    def crop_size(p):
        dpi = p.dpi if use_dpi else 1.0
        pad = 2 * (output_margin_mm / 25.4)
        # uniform size, but never smaller than THIS page's own content+margin
        # (so an outlier page is never clipped -- it just gets a larger crop)
        w_in = max(uni_w, extent(p, p.margin.content.width)) + pad
        h_in = max(uni_h, extent(p, p.margin.content.height)) + pad
        cw = w_in * (dpi if use_dpi else 1)
        ch = h_in * (dpi if use_dpi else 1)
        return cw, ch

    def baseline(p):  # this page's own content, centered in the uniform crop
        c = p.margin.content
        cw, ch = crop_size(p)
        return c.x0 + c.width / 2 - cw / 2, c.y0 + c.height / 2 - ch / 2

    # Common nombre CORRECTION, relative to the centered baseline (not to
    # content's edge, so the correction's mean is ~0 and doesn't drag pages
    # off-center). Horizontal is computed PER PARITY: recto/verso legitimately
    # place the nombre on opposite sides. Vertical is computed ACROSS BOTH
    # PARITIES TOGETHER: a real book prints the nombre at the SAME height
    # regardless of recto/verso -- measured on real corpus data, raw-scan
    # nombre.y0 is ~2371 for both odd and even pages (no parity-dependent
    # offset at all). Splitting the vertical anchor by parity (as it was
    # first written, by analogy with the horizontal one) let the two
    # parities' baselines drift apart and reintroduced an ~80px recto/verso
    # height mismatch that doesn't exist in the source.
    common_noff_x: dict[int, float] = {}
    by_parity: dict[int, list] = {0: [], 1: []}
    for p in have:
        by_parity[p.source.page_index % 2].append(p)
    for parity, group in by_parity.items():
        offs_x = [extent(p, p.margin.nombre_box.x0 - baseline(p)[0])
                  for p in group if confident(p)]
        if offs_x:
            common_noff_x[parity] = float(np.median(offs_x))

    offs_y = [extent(p, p.margin.nombre_box.y0 - baseline(p)[1])
              for p in have if confident(p)]
    common_noff = float(np.median(offs_y)) if offs_y else None

    for p in have:
        dpi = p.dpi if use_dpi else 1.0
        margin_px = output_margin_mm / 25.4 * (p.dpi if use_dpi else 96)
        crop_w, crop_h = crop_size(p)
        bx0, by0 = baseline(p)
        c = p.margin.content
        noff_x = common_noff_x.get(p.source.page_index % 2)
        if confident(p) and noff_x is not None:
            # centered baseline + the common per-parity nombre correction
            x0 = p.margin.nombre_box.x0 - noff_x * (dpi if use_dpi else 1)
        else:
            x0 = bx0                                  # fallback: centered
        if confident(p) and common_noff is not None:
            y0 = p.margin.nombre_box.y0 - common_noff * (dpi if use_dpi else 1)
        else:
            y0 = by0                                   # fallback: centered
        # clamp so EVERY side keeps >= the output margin (never touches content);
        # always feasible since the uniform crop >= content + 2*margin
        x0 = max(min(x0, c.x0 - margin_px), c.x1 + margin_px - crop_w)
        y0 = max(min(y0, c.y0 - margin_px), c.y1 + margin_px - crop_h)
        p.margin.crop = Box(x0, y0, x0 + crop_w, y0 + crop_h)


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
