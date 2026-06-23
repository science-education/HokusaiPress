"""Mixed Raster Content (MRC) page assembly.

A page with photos/figures should not be stored as one big JPEG (text turns
soft, file gets heavy) nor as one bilevel image (halftones turn to mud).
MRC splits each page into layers in a single PDF page:

- base layer: the whole page binarized to crisp G4/JBIG2, with photo regions
  erased to white so halftone never gets binarized;
- photo layers: each photo region cropped from the full-resolution render and
  down-sampled to a modest dpi, encoded as JPEG and overlaid at its location;
- text layer: the invisible searchable layer.

This keeps characters razor-sharp and light while photos stay legible at a
fraction of the bytes. All validated encoders are reused from hybrid-ocr's
pdf_export; this module only composes their single-page PDFs with pikepdf.

Compositing is done in pixel space (1px = 1pt); the caller rescales MediaBoxes
to physical points afterwards.
"""

from __future__ import annotations

from io import BytesIO
import zlib

import cv2
import numpy as np

# Tint-zone local thresholding constants.
# Text strokes on a tint background are pixels darker than the LOCAL
# background estimate by at least this many grey levels.  Tuned for
# ADF-scanned halftone panels where the background peaks near 165-180 and
# text strokes are at 60-130.
_TINT_LO: int = 60
_TINT_HI: int = 220
_TINT_TEXT_DELTA: int = 40  # local_bg(tint pix) - 40 → local ink threshold

# Local-background window radius (see _local_tint_background).  A single
# TintZone can now span most of the page (header+column+footer merged into
# one shape -- see tint_zone.py), so one median over the whole zone is too
# coarse: a darker accent area (e.g. a title printed on a deliberately
# darker strip of the header) sits close enough to the page-wide median that
# ordinary scan grain straddles the resulting threshold, speckling that area
# black/white at the pixel level.
#
# The estimator is a MEDIAN filter, not morphological closing: closing
# (dilate-then-erode) is biased toward the brighter extreme near any bright
# feature -- right around a white knockout letter, dilate spreads the
# letter's brightness outward and erode needs an even wider dark margin to
# pull it back down, so the background estimate just outside the letter
# reads brighter than the true tint.  That inflates ink_T enough to
# misclassify genuine tint-background pixels near the letter as ink, drawn
# as a dark halo hugging every stroke.  A median is symmetric: as long as
# background pixels are the majority inside the window (true away from
# unusually dense text), it tracks the true local tint regardless of nearby
# bright or dark outliers.
#
# 25px is wide enough to outvote ordinary body-text-scale stroke width
# without blurring across genuine background-tone transitions at that scale.
# Small, mostly-decorative zones (e.g. a circular badge with its own small
# caption, detected as its own zone because it isn't physically connected to
# a larger panel) use a smaller window: a badge's own lettering is small
# enough that a 25px window would itself be dominated by the letters' own
# white knockout rather than the surrounding tint.
_LOCAL_BG_WINDOW_PX: int = 25
_LOCAL_BG_WINDOW_SMALL_PX: int = 10
_LOCAL_BG_SMALL_ZONE_DIM: int = 600  # zones with both dims <= this use the small window
_LOCAL_BG_MAX_DIM: int = 2400  # downscale ceiling so filtering a page-sized zone stays cheap

# Tint overlays downsample less aggressively than photo overlays
# (self.scale, tied to photo_dpi).  A real photograph's own texture hides
# JPEG block boundaries; a tint panel is a large flat colour field, so the
# SAME amount of downsampling makes 8x8 JPEG blocks clearly visible once
# upsampled back to page resolution -- a "mosaic" that doesn't exist in the
# source scan.  300 dpi (vs photo_dpi's default 200) costs roughly +30% on
# this layer's bytes but removes the visible blocking.
_TINT_OVERLAY_DPI: int = 300

# Posterized lossless tint overlays.  These are only used for non-photo tint
# panels and only when the measured error against the source overlay stays low.
_TINT_POSTERIZE_K: int = 4
_TINT_POSTERIZE_SAMPLE_MAX: int = 200_000
_TINT_POSTERIZE_MAE_MAX: float = 10.0
_TINT_POSTERIZE_P95_MAX: float = 28.0
_TINT_POSTERIZE_BAD_DELTA: int = 32
_TINT_POSTERIZE_BAD_FRAC_MAX: float = 0.03
# Text strokes (black ink AND white knockout) detected over a tint zone are
# punched out of the posterized overlay as 255 "holes": the bilevel base layer
# below already carries the glyph (black strokes / white paper), and a Multiply
# overlay treats 255 as identity, so the base shows through unmodified.  Feeding
# those glyph pixels -- and especially their anti-aliased edges -- into the k4
# clustering instead pulls a stray light/dark cluster that speckles the tint
# background around the text.  The dilation grabs the anti-aliased halo so the
# background quantizer only ever sees clean tint.
_TINT_TEXT_HOLE_DILATE_PX: int = 2

# The PDF tint overlay is clipped to the exact contour, but the bilevel base
# layer should be a little more forgiving at the OUTER tint boundary.  ADF/Otsu
# can leave a tiny black edge blob just outside the contour; whitening the base
# under a slightly dilated outer mask removes that without changing the overlay
# geometry.  Holes are not dilated, so body text inside a frame cavity is not
# eaten.
_BASE_WHITEOUT_DILATE_PX: int = 2
_TINT_EDGE_DIRT_BAND_PX: int = 8

# After thresholding, an isolated 1-2px "ink" speck (ordinary scan grain that
# dipped just below ink_T) is removed: a real text stroke is always several
# pixels wide and connected, so opening with a kernel below stroke width
# only erases noise that was never a real stroke to begin with.
_INK_DESPECKLE_PX: int = 1

# White knockout text / paper cutouts inside a tint zone are deliberately NOT
# ink.  Scanner noise in those bright areas can be very dark and can form
# clusters too large for the final page-level despeckle.  OCR text boxes let
# us distinguish "text for search" from "black text for bilevel rendering":
# a bright text box on a tint background is rendered by the tint overlay, not
# by the bilevel ink mask.
_KNOCKOUT_WHITE_MIN: int = 225
_KNOCKOUT_DILATE_PX: int = 2
_KNOCKOUT_NOISE_MAX_AREA: int = 512
_KNOCKOUT_TEXT_BRIGHT_FRAC_MIN: float = 0.02
_KNOCKOUT_TEXT_DARK_FRAC_MAX: float = 0.12
_KNOCKOUT_TEXT_PAD_PX: int = 2

# Phase 2 -- gray text that is neither clearly black ink nor bright knockout.
# An OCR box whose strokes sit a clear distance from the box's own background
# tone (but not down at ink black or up at knockout white) is mid-gray text.
# Its strokes are punched out of the tint background quantizer like any other
# text, but the hole is filled with the ESTIMATED stroke gray instead of paper
# white, and the bilevel base under it is whitened so the gray overlay shows.
# Detection is deliberately conservative: a real contrast (>= _DELTA) over a
# text-like, not tone-patch-like, fraction of the box.
_TINT_GRAY_TEXT_DELTA: int = 40
_TINT_GRAY_TEXT_FRAC_MIN: float = 0.02
_TINT_GRAY_TEXT_FRAC_MAX: float = 0.45

# Hysteresis for tint-zone ink.  Weak pixels are kept only when connected to a
# stronger core, which prevents ordinary tint grain from becoming a black
# component while preserving real strokes that have a dark centre.
_TINT_STRONG_TEXT_DELTA: int = 65
_TINT_STRONG_ABS_MAX: int = 105

# A tint tone transition (for example the lower-right corner of a dark header
# block meeting a lighter band) can make halftone dots connect into a compact
# component.  It passes the weak/strong hysteresis, but unlike real black text
# it has no genuinely dark core and it sits in a high-contrast neighbourhood.
_TINT_EDGE_NOISE_MAX_AREA: int = 192
_TINT_EDGE_NOISE_DARK_CORE_MAX: int = 95
_TINT_EDGE_NOISE_NEIGHBOR_PX: int = 8
_TINT_EDGE_NOISE_NEIGHBOR_RANGE_MIN: int = 80

# The GLOBAL Otsu threshold (binarize_bw) has no despeckling at all -- only
# the tint-zone-internal ink mask above does.  Ordinary scan grain is an
# invisible few grey levels of noise in the continuous-tone source, but a
# hard threshold amplifies an unlucky noise dip into a stark, fully black
# pixel: thresholding makes invisible noise visible.  This shows up as
# isolated black flecks scattered near any tint/paper edge (where the global
# threshold sits closest to the local pixel values) and is not specific to
# tint zones -- it's a property of binarizing any noisy scan.  A connected-
# component area filter removes specks below this size; a real character
# stroke is always much larger and connected, so this never touches text.
_GLOBAL_DESPECKLE_MIN_PX: int = 8


class MrcPageBuilder:
    """Accumulates pages and writes one searchable MRC PDF."""

    def __init__(self, compress: str = "g4", photo_dpi: int = 200,
                 target_dpi: int = 600, jpeg_quality: int = 85,
                 ink_valley: "int | None" = None):
        self.compress = compress
        self.scale = max(photo_dpi / target_dpi, 0.05)
        self.tint_scale = max(_TINT_OVERLAY_DPI / target_dpi, 0.05)
        self.jpeg_quality = jpeg_quality
        self.ink_valley = ink_valley   # 2-pass book binarization valley (or None)
        self._pages: list[dict] = []

    def add_page(self, out_bgr: np.ndarray, lines: list,
                 photo_boxes_px: list, mode: str,
                 vector_fills: list | None = None,
                 tint_zones: list | None = None) -> None:
        from hybrid_ocr.pdf_export import encode_page_pdf

        from .render import binarize_bw

        h, w = out_bgr.shape[:2]
        vector_fills = vector_fills or []
        tint_zones = tint_zones or []
        tint_overlays: list[dict] = []
        vector_tint_fills: list[dict] = []

        if mode in ("gray", "color"):
            base_pdf = encode_page_pdf(out_bgr, mode, self.compress)
            overlays = []
        else:
            binary = binarize_bw(out_bgr, self.ink_valley)  # {0,255}, 2-pass valley
            overlays = []
            for box in photo_boxes_px:
                x0, y0, x1, y1 = box[:4]
                tone = box[4] if len(box) > 4 else None  # None|"gray"|"color"
                x0i, y0i = max(0, int(x0)), max(0, int(y0))
                x1i, y1i = min(w, int(x1)), min(h, int(y1))
                # guard on the SCALED size: a region that's wide enough at full
                # res can still downsample to a 1px sliver, which img2pdf/pikepdf
                # rejects as a degenerate page (< 3 PDF units at assumed DPI).
                ds_w = max(1, int((x1i - x0i) * self.scale))
                ds_h = max(1, int((y1i - y0i) * self.scale))
                if x1i - x0i < 4 or y1i - y0i < 4 or ds_w < 4 or ds_h < 4:
                    continue
                binary[y0i:y1i, x0i:x1i] = 255   # erase photo from bilevel
                crop = out_bgr[y0i:y1i, x0i:x1i]
                ds = cv2.resize(crop, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
                # explicit per-region override wins; else decide from chroma
                cmode = tone or ("gray" if _is_grayish(crop) else "color")
                overlays.append({
                    "pdf": encode_page_pdf(ds, cmode, self.compress),
                    "rect": (x0i, h - y1i, x1i, h - y0i),  # PDF y-up
                })
            # Tint zones: place a greyscale Multiply-blend overlay that preserves
            # the per-pixel tint colour.  Text strokes are identified with a
            # local background threshold (local tint estimate - _TINT_TEXT_DELTA)
            # rather than the global Otsu threshold or a single whole-zone median.
            #
            # Why not global Otsu?  The page-level Otsu is driven by the large
            # white-paper peak (mode ~240) and sets T ≈ 200-215, which binarises
            # the entire tint background as black -- exactly wrong for a
            # Multiply overlay.
            #
            # Why not one median for the whole zone?  A zone can now span most
            # of the page (see tint_zone.py), so a single median averages over
            # areas with genuinely different background tone; see
            # _local_tint_background for the per-pixel alternative.
            for zone in tint_zones:
                x0, y0, x1, y1 = zone.rect
                x0i, y0i = max(0, int(x0)), max(0, int(y0))
                x1i, y1i = min(w, int(x1)), min(h, int(y1))
                if x1i - x0i < 4 or y1i - y0i < 4:
                    continue
                crop_gray = cv2.cvtColor(
                    out_bgr[y0i:y1i, x0i:x1i], cv2.COLOR_BGR2GRAY
                )
                # Build polygon mask: only modify pixels inside the contour shape.
                # Holes (e.g. the body-text cavity enclosed by a header/column/
                # footer frame) are punched out so they're left untouched.
                zone_h, zone_w = y1i - y0i, x1i - x0i
                if zone.page_contour is not None:
                    outer_mask = np.zeros((zone_h, zone_w), dtype=np.uint8)
                    cnt_crop = zone.page_contour.copy()
                    cnt_crop[:, 0, 0] -= x0i
                    cnt_crop[:, 0, 1] -= y0i
                    cv2.fillPoly(outer_mask, [cnt_crop], 255)
                    hole_mask = np.zeros((zone_h, zone_w), dtype=np.uint8)
                    for hole in zone.hole_contours:
                        hole_crop = hole.copy()
                        hole_crop[:, 0, 0] -= x0i
                        hole_crop[:, 0, 1] -= y0i
                        cv2.fillPoly(hole_mask, [hole_crop], 255)
                    in_shape = (outer_mask > 0) & (hole_mask == 0)
                    base_shape = _base_whiteout_mask(outer_mask, hole_mask)
                else:
                    outer_mask = np.full((zone_h, zone_w), 255, dtype=np.uint8)
                    hole_mask = np.zeros((zone_h, zone_w), dtype=np.uint8)
                    in_shape = np.ones((zone_h, zone_w), dtype=bool)
                    base_shape = in_shape

                tint_pix = crop_gray[
                    (crop_gray >= _TINT_LO) & (crop_gray <= _TINT_HI) & in_shape
                ]
                edge_fill_value = (
                    int(round(float(np.median(tint_pix)))) if tint_pix.size else 255
                )
                if tint_pix.size >= 100:
                    # The zone's bounding RECT is not the zone's own shape --
                    # a rounded corner or circle's bbox always includes some
                    # paper-coloured corner area outside the actual polygon.
                    # Feeding that paper straight into the median window would
                    # pull the local-background estimate toward white right
                    # along the true tint/paper edge, the same way a bright
                    # knockout letter does (see _local_tint_background) --
                    # producing the same kind of false "ink" speck, but
                    # tracing the zone's own outline instead of a letterform.
                    # Filling out-of-shape pixels with the zone's own tint
                    # median keeps that real edge from leaking into the
                    # window used near it.
                    masked_crop = np.where(
                        in_shape, crop_gray, edge_fill_value
                    ).astype(np.uint8)
                    knockout_text = _knockout_text_mask(
                        crop_gray, lines, x0i, y0i, in_shape
                    )
                    ink = _tint_ink_mask(masked_crop, knockout_text)
                else:
                    ink = binary[y0i:y1i, x0i:x1i] == 0  # fallback
                edge_dirt = _tint_overlay_edge_dirt_mask(crop_gray, outer_mask, hole_mask)
                ink[edge_dirt] = 0
                ink_in_shape = ink & in_shape
                # Locate every text stroke (black ink / white knockout / mid-
                # gray) so it can be punched out of the posterized overlay's
                # background quantizer instead of speckling the tint around it.
                text_hole, text_fill, gray_text = _tint_text_holes(
                    crop_gray, lines, x0i, y0i, in_shape, ink_in_shape
                )
                region = binary[y0i:y1i, x0i:x1i]
                region[base_shape & ~ink_in_shape] = 255  # tint bg / edge → white
                region[ink_in_shape] = 0                  # text strokes → black
                # Mid-gray glyphs are carried by the gray overlay fill, not the
                # bilevel base: whiten the base so the Multiply overlay shows.
                if gray_text.any():
                    region[gray_text] = 255
                if zone.page_contour is not None:
                    fx0, fy0, fringe = _base_edge_fringe_mask(
                        zone.page_contour, zone.hole_contours, h, w
                    )
                    if fringe.size:
                        edge_region = binary[fy0:fy0 + fringe.shape[0],
                                             fx0:fx0 + fringe.shape[1]]
                        edge_region[fringe] = 255
                # Overlay: greyscale, downsampled to _TINT_OVERLAY_DPI (less
                # aggressively than photo overlays -- see that constant).
                #
                # Do not reduce tint zones to PDF vector fills here.  Large
                # halftone areas can be semantically flat, but clipped vector
                # fills make seams visible at tone boundaries and glyph holes.
                # Keep tint as raster and investigate posterized lossless
                # image compression instead.
                ov_gray = zone.overlay_img              # HxW uint8
                ov_gray = _clean_tint_overlay_edge(ov_gray, edge_dirt, edge_fill_value)
                ov_ds = cv2.resize(
                    ov_gray,
                    # pikepdf/img2pdf both reject PDF page sizes below 3 units,
                    # and this 1px=1pt downsampled overlay becomes its own
                    # mini-PDF page -- floor at 3, not 1, so a thin tint
                    # strip (e.g. a hairline shadow band) can't crash either
                    # codec path (k4_flate or the jpeg/img2pdf fallback).
                    (max(3, int((x1i - x0i) * self.tint_scale)),
                     max(3, int((y1i - y0i) * self.tint_scale))),
                    interpolation=cv2.INTER_AREA,
                )
                in_shape_ds = cv2.resize(
                    in_shape.astype(np.uint8),
                    (ov_ds.shape[1], ov_ds.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
                ink_ds = cv2.resize(
                    ink_in_shape.astype(np.uint8),
                    (ov_ds.shape[1], ov_ds.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
                hole_ds = cv2.resize(
                    text_hole.astype(np.uint8),
                    (ov_ds.shape[1], ov_ds.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
                hole_fill_ds = cv2.resize(
                    text_fill,
                    (ov_ds.shape[1], ov_ds.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                posterized = _try_posterized_tint_overlay_pdf(
                    ov_ds, in_shape_ds, ink_ds, hole_ds, hole_fill_ds
                )
                if posterized is not None:
                    overlay_pdf, poster_stats = posterized
                else:
                    overlay_pdf = encode_page_pdf(
                        cv2.cvtColor(ov_ds, cv2.COLOR_GRAY2BGR), "gray", self.compress
                    )
                    poster_stats = {"codec": "jpeg_fallback"}
                tint_overlays.append({
                    "pdf": overlay_pdf,
                    "rect": (x0i, h - y1i, x1i, h - y0i),  # PDF y-up
                    "contour": zone.page_contour,
                    "holes": zone.hole_contours,
                    "circle_fit": zone.circle_fit,
                    "page_h_px": h,
                    "posterize": poster_stats,
                })
            for fill in vector_fills:
                x0, y0, x1, y1 = fill["rect"]
                x0i, y0i = max(0, int(x0)), max(0, int(y0))
                x1i, y1i = min(w, int(x1)), min(h, int(y1))
                if x1i - x0i < 1 or y1i - y0i < 1:
                    continue
                crop = out_bgr[y0i:y1i, x0i:x1i]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                ink = gray < max(0, int(round(fill["gray"] * 255)) - 50)
                ink[_rect_edge_dirt_mask(gray)] = False
                binary[y0i:y1i, x0i:x1i] = 255
                region = binary[y0i:y1i, x0i:x1i]
                region[ink] = 0
            binary = _despeckle_bilevel(binary)
            base_pdf = encode_page_pdf(binary, "bw", self.compress)

        self._pages.append({
            "base": base_pdf,
            "overlays": overlays,
            "tint_overlays": tint_overlays,
            "vector_tint_fills": vector_tint_fills,
            "vector_fills": vector_fills,
            "w": w,
            "h": h,
            "lines": lines,
        })

    def save(self, output_path: str) -> None:
        import pikepdf
        from hybrid_ocr.pdf_export import build_text_overlay

        if not self._pages:
            raise ValueError("no pages added")

        text_pdf_bytes = build_text_overlay(
            [(p["w"], p["h"], p["lines"]) for p in self._pages]
        )
        out = pikepdf.new()
        sources = []
        try:
            text_pdf = pikepdf.open(BytesIO(text_pdf_bytes))
            sources.append(text_pdf)
            for i, page in enumerate(self._pages):
                base = pikepdf.open(BytesIO(page["base"]))
                sources.append(base)
                out.pages.extend(base.pages)
                dest = out.pages[-1]
                for j, ov in enumerate(page["overlays"]):
                    ovpdf = pikepdf.open(BytesIO(ov["pdf"]))
                    sources.append(ovpdf)
                    _add_overlay_named(
                        dest,
                        ovpdf.pages[0],
                        pikepdf.Rectangle(*ov["rect"]),
                        f"/HPPhoto{i}_{j}",
                    )
                if page.get("tint_overlays"):
                    _add_tint_overlays(out, dest, page["tint_overlays"], i, sources)
                if page.get("vector_tint_fills"):
                    _add_vector_tint_fills(out, dest, page["vector_tint_fills"], page["h"])
                if page["vector_fills"]:
                    _add_vector_fills(out, dest, page["vector_fills"], page["h"])
                _add_overlay_named(dest, text_pdf.pages[i], None, f"/HPText{i}")
            out.save(output_path, deterministic_id=True)
        finally:
            for s in sources:
                s.close()


def _despeckle_bilevel(binary: np.ndarray) -> np.ndarray:
    """Remove isolated ink specks below _GLOBAL_DESPECKLE_MIN_PX from a
    bilevel ({0, 255}) page.  See that constant's docstring for why this is
    needed even after Otsu: thresholding turns invisible scan grain into
    stark black flecks, most visibly near tint/paper edges and any other
    boundary where pixel values already sit close to the threshold.
    """
    ink = (binary == 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
    tiny = np.flatnonzero(sizes < _GLOBAL_DESPECKLE_MIN_PX) + 1
    if tiny.size:
        out = binary.copy()
        out[np.isin(labels, tiny)] = 255
        return out
    return binary


def _is_grayish(bgr: np.ndarray) -> bool:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    chroma = lab[..., 1].astype(np.float32).std() + lab[..., 2].astype(np.float32).std()
    return chroma <= 16.0


def _base_whiteout_mask(outer_mask: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    """Bilevel-base whitening mask for a tint zone.

    The overlay is clipped to the exact contour, but the base layer is whitened
    under a slightly dilated OUTER shape so Otsu-created black edge flecks just
    outside the contour do not survive.  Structural holes are subtracted after
    dilation, keeping body-text cavities untouched.
    """
    if _BASE_WHITEOUT_DILATE_PX <= 0:
        grown = outer_mask > 0
    else:
        k = 2 * _BASE_WHITEOUT_DILATE_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        grown = cv2.dilate(outer_mask, kernel, iterations=1) > 0
    return grown & (hole_mask == 0)


def _base_edge_fringe_mask(
    page_contour: np.ndarray,
    hole_contours: list,
    page_h: int,
    page_w: int,
) -> tuple[int, int, np.ndarray]:
    """Outer-edge-only part of _base_whiteout_mask in page coordinates."""
    if _BASE_WHITEOUT_DILATE_PX <= 0 or page_contour is None or len(page_contour) < 3:
        return 0, 0, np.zeros((0, 0), dtype=bool)

    bx, by, bw, bh = cv2.boundingRect(page_contour)
    pad = _BASE_WHITEOUT_DILATE_PX + 2
    x0 = max(0, bx - pad)
    y0 = max(0, by - pad)
    x1 = min(page_w, bx + bw + pad)
    y1 = min(page_h, by + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, np.zeros((0, 0), dtype=bool)

    outer = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cnt = page_contour.copy()
    cnt[:, 0, 0] -= x0
    cnt[:, 0, 1] -= y0
    cv2.fillPoly(outer, [cnt], 255)

    holes = np.zeros_like(outer)
    for hole in hole_contours or []:
        hc = hole.copy()
        hc[:, 0, 0] -= x0
        hc[:, 0, 1] -= y0
        cv2.fillPoly(holes, [hc], 255)

    in_shape = (outer > 0) & (holes == 0)
    whiteout = _base_whiteout_mask(outer, holes)
    return x0, y0, whiteout & ~in_shape


def _tint_overlay_edge_dirt_mask(
    gray: np.ndarray,
    outer_mask: np.ndarray,
    hole_mask: np.ndarray,
) -> np.ndarray:
    in_shape = (outer_mask > 0) & (hole_mask == 0)
    if not in_shape.any():
        return np.zeros_like(gray, dtype=bool)

    k = 2 * _TINT_EDGE_DIRT_BAND_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(in_shape.astype(np.uint8), kernel, iterations=1) > 0
    boundary = in_shape & ~eroded
    return (gray <= _TINT_STRONG_ABS_MAX) & boundary


def _clean_tint_overlay_edge(
    overlay_gray: np.ndarray,
    edge_dirt: np.ndarray,
    fill_value: int,
) -> np.ndarray:
    """Replace dark overlay dirt in the outer tint boundary band."""
    if not edge_dirt.any():
        return overlay_gray
    out = overlay_gray.copy()
    fill = np.uint8(max(0, min(255, int(fill_value))))
    out[edge_dirt] = fill
    return out


def _rect_edge_dirt_mask(gray: np.ndarray) -> np.ndarray:
    """Dark boundary artifacts for rectangular vector-fill regions."""
    h, w = gray.shape[:2]
    if h < 1 or w < 1:
        return np.zeros_like(gray, dtype=bool)
    band = max(1, _TINT_EDGE_DIRT_BAND_PX)
    edge = np.zeros((h, w), dtype=bool)
    edge[:band, :] = True
    edge[max(0, h - band):, :] = True
    edge[:, :band] = True
    edge[:, max(0, w - band):] = True
    return (gray <= _TINT_STRONG_ABS_MAX) & edge


def _local_tint_background(crop_gray: np.ndarray) -> np.ndarray:
    """Per-pixel local background estimate via a median filter.

    See _LOCAL_BG_WINDOW_PX docstring for why a median (not morphological
    closing) is used.  Window radius is picked from the crop's own size: an
    earlier content-based estimate (measuring stroke width via a coarse ink
    mask) was tried and discarded -- for a zone whose crop spans most of the
    page, the dominant "ink" by area is ordinary body-paragraph text, which
    has nothing to do with the stroke scale actually relevant to the tint
    panel embedded in that same crop, and using it skewed the window badly.
    Downscaled for very large zones so filtering a page-sized crop stays cheap.
    """
    h, w = crop_gray.shape[:2]
    # Estimate the background of the tint itself.  Bright knockout letters and
    # paper corners must not pull the median upward, and black strokes must not
    # pull it downward; both are replaced by the zone's own tint median before
    # the local median filter runs.  This keeps the estimate tied to "what the
    # tint would be here", not to the visible foreground currently occupying
    # the pixel.
    tint = (crop_gray >= _TINT_LO) & (crop_gray <= _TINT_HI)
    if int(tint.sum()) >= 4:
        fill_value = int(round(float(np.median(crop_gray[tint]))))
        work_src = np.where(tint, crop_gray, fill_value).astype(np.uint8)
    else:
        work_src = crop_gray

    radius = (
        _LOCAL_BG_WINDOW_SMALL_PX
        if max(h, w) <= _LOCAL_BG_SMALL_ZONE_DIM
        else _LOCAL_BG_WINDOW_PX
    )

    max_dim = max(h, w)
    if max_dim > _LOCAL_BG_MAX_DIM:
        scale = _LOCAL_BG_MAX_DIM / float(max_dim)
        work = cv2.resize(
            work_src,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        radius = max(1, int(round(radius * scale)))
    else:
        work = work_src

    k = 2 * radius + 1
    # cv2.medianBlur only accepts uint8/float32 with ksize <= 5 for multi-
    # channel; for single-channel uint8 any odd ksize is fine.
    bg = cv2.medianBlur(work, k)
    if work.shape != crop_gray.shape:
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return bg.astype(np.float32)


def _knockout_text_mask(
    crop_gray: np.ndarray,
    lines: list,
    crop_x0: int,
    crop_y0: int,
    in_shape: np.ndarray,
) -> np.ndarray:
    """Mask OCR text boxes that are bright knockout text, not black ink.

    OCR text remains searchable through the invisible text layer.  This mask is
    only about whether a text box should contribute black pixels to the bilevel
    base layer.  A box with visible white glyph pixels and little dark ink is a
    knockout label on a tint background, so its whole box is left to the tint
    overlay instead of being thresholded into black.
    """
    h, w = crop_gray.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    for line in lines:
        box = line.get("box") if isinstance(line, dict) else None
        if not box or len(box) < 4:
            continue
        x0f, y0f, x1f, y1f = (float(v) for v in box[:4])
        x0 = max(0, int(np.floor(min(x0f, x1f))) - crop_x0)
        y0 = max(0, int(np.floor(min(y0f, y1f))) - crop_y0)
        x1 = min(w, int(np.ceil(max(x0f, x1f))) - crop_x0)
        y1 = min(h, int(np.ceil(max(y0f, y1f))) - crop_y0)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue

        valid = in_shape[y0:y1, x0:x1]
        valid_px = int(valid.sum())
        if valid_px < 8:
            continue
        sub = crop_gray[y0:y1, x0:x1]
        bright_frac = float(((sub >= _KNOCKOUT_WHITE_MIN) & valid).sum()) / valid_px
        dark_frac = float(((sub <= _TINT_STRONG_ABS_MAX) & valid).sum()) / valid_px
        if (
            bright_frac >= _KNOCKOUT_TEXT_BRIGHT_FRAC_MIN
            and dark_frac <= _KNOCKOUT_TEXT_DARK_FRAC_MAX
        ):
            x0p = max(0, x0 - _KNOCKOUT_TEXT_PAD_PX)
            y0p = max(0, y0 - _KNOCKOUT_TEXT_PAD_PX)
            x1p = min(w, x1 + _KNOCKOUT_TEXT_PAD_PX)
            y1p = min(h, y1 + _KNOCKOUT_TEXT_PAD_PX)
            out[y0p:y1p, x0p:x1p] |= in_shape[y0p:y1p, x0p:x1p]
    return out


def _ocr_text_box_mask(
    crop_gray: np.ndarray,
    lines: list,
    crop_x0: int,
    crop_y0: int,
    in_shape: np.ndarray,
) -> np.ndarray:
    """Mask OCR text boxes for local raster fallback over vector tint fills."""
    h, w = crop_gray.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    pad = 4
    for line in lines:
        box = line.get("box") if isinstance(line, dict) else None
        if not box or len(box) < 4:
            continue
        x0f, y0f, x1f, y1f = (float(v) for v in box[:4])
        x0 = max(0, int(np.floor(min(x0f, x1f))) - crop_x0 - pad)
        y0 = max(0, int(np.floor(min(y0f, y1f))) - crop_y0 - pad)
        x1 = min(w, int(np.ceil(max(x0f, x1f))) - crop_x0 + pad)
        y1 = min(h, int(np.ceil(max(y0f, y1f))) - crop_y0 + pad)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        sub = crop_gray[y0:y1, x0:x1]
        valid = in_shape[y0:y1, x0:x1]
        valid_px = int(valid.sum())
        if valid_px < 8:
            continue
        bright_frac = float(((sub >= _KNOCKOUT_WHITE_MIN) & valid).sum()) / valid_px
        dark_frac = float(((sub <= _TINT_STRONG_ABS_MAX) & valid).sum()) / valid_px
        if not (
            bright_frac >= _KNOCKOUT_TEXT_BRIGHT_FRAC_MIN
            and dark_frac <= _KNOCKOUT_TEXT_DARK_FRAC_MAX
        ):
            continue
        text_pixels = (sub >= _KNOCKOUT_WHITE_MIN) & valid
        if int(text_pixels.sum()) < 4:
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        text_pixels = cv2.dilate(text_pixels.astype(np.uint8), kernel, iterations=1) > 0
        out[y0:y1, x0:x1] |= text_pixels & valid
    return out


def _tint_text_holes(
    crop_gray: np.ndarray,
    lines: list,
    crop_x0: int,
    crop_y0: int,
    in_shape: np.ndarray,
    ink: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate text strokes over a tint zone and the colour to render them.

    The posterized tint overlay is Multiply-blended over the bilevel base, so a
    text stroke must be *punched out* of the background quantizer rather than
    clustered with it -- otherwise the stroke and its anti-aliased edge seed a
    stray light/dark cluster that speckles the tint around the text.

    Returns ``(hole_mask, hole_fill, gray_text_mask)`` at the crop resolution:

    * ``hole_mask``  -- every stroke pixel (black ink, white knockout, mid-gray)
      plus a small dilation for the anti-aliased halo.  Excluded from the k4
      background clustering and overwritten in the overlay.
    * ``hole_fill``  -- the overlay value to write at hole pixels.  255 (paper)
      for black ink and white knockout, since the base layer already carries the
      glyph; the estimated stroke gray for mid-gray text, which the overlay must
      paint itself.
    * ``gray_text_mask`` -- mid-gray strokes whose base must be whitened so the
      gray overlay fill is visible through the Multiply blend.
    """
    h, w = crop_gray.shape[:2]
    hole = np.zeros((h, w), dtype=bool)
    gray_text = np.zeros((h, w), dtype=bool)
    fill = np.full((h, w), 255, dtype=np.uint8)

    dk = 2 * _TINT_TEXT_HOLE_DILATE_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))

    # Black ink: holes filled with paper white -- the base layer draws the black
    # stroke, and 255 is the Multiply identity, so it shows through unchanged.
    # The dilation pulls the anti-aliased stroke edge out of the background too.
    ink_in = ink & in_shape
    if ink_in.any():
        ink_halo = (cv2.dilate(ink_in.astype(np.uint8), kernel) > 0) & in_shape
        hole |= ink_halo

    pad = _KNOCKOUT_TEXT_PAD_PX
    for line in lines:
        box = line.get("box") if isinstance(line, dict) else None
        if not box or len(box) < 4:
            continue
        x0f, y0f, x1f, y1f = (float(v) for v in box[:4])
        x0 = max(0, int(np.floor(min(x0f, x1f))) - crop_x0 - pad)
        y0 = max(0, int(np.floor(min(y0f, y1f))) - crop_y0 - pad)
        x1 = min(w, int(np.ceil(max(x0f, x1f))) - crop_x0 + pad)
        y1 = min(h, int(np.ceil(max(y0f, y1f))) - crop_y0 + pad)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        sub = crop_gray[y0:y1, x0:x1]
        valid = in_shape[y0:y1, x0:x1]
        valid_px = int(valid.sum())
        if valid_px < 8:
            continue
        bright = (sub >= _KNOCKOUT_WHITE_MIN) & valid
        darkabs = (sub <= _TINT_STRONG_ABS_MAX) & valid
        bright_frac = float(bright.sum()) / valid_px
        dark_frac = float(darkabs.sum()) / valid_px

        # White knockout glyphs: stroke pixels filled with paper (base is white).
        if (
            bright_frac >= _KNOCKOUT_TEXT_BRIGHT_FRAC_MIN
            and dark_frac <= _KNOCKOUT_TEXT_DARK_FRAC_MAX
            and int(bright.sum()) >= 4
        ):
            strokes = cv2.dilate(bright.astype(np.uint8), kernel) > 0
            hole[y0:y1, x0:x1] |= strokes & valid
            continue

        # Mid-gray text: strokes sit a clear distance from the box's own tint
        # background but are neither ink-black nor knockout-white.  Estimate the
        # stroke colour and render it directly from the overlay.
        bg = float(np.median(sub[valid]))
        gdark = (sub.astype(np.int16) <= bg - _TINT_GRAY_TEXT_DELTA) & valid & ~darkabs
        gbright = (sub.astype(np.int16) >= bg + _TINT_GRAY_TEXT_DELTA) & valid & ~bright
        gdark_frac = float(gdark.sum()) / valid_px
        gbright_frac = float(gbright.sum()) / valid_px
        if gdark_frac >= gbright_frac:
            cand, cfrac = gdark, gdark_frac
        else:
            cand, cfrac = gbright, gbright_frac
        if not (_TINT_GRAY_TEXT_FRAC_MIN <= cfrac <= _TINT_GRAY_TEXT_FRAC_MAX):
            continue
        if int(cand.sum()) < 4:
            continue
        color = int(round(float(np.median(sub[cand]))))
        strokes = (cv2.dilate(cand.astype(np.uint8), kernel) > 0) & valid
        # Do not override a black-ink hole that the dilation may reach into.
        sub_hole = hole[y0:y1, x0:x1]
        new_gray = strokes & ~sub_hole
        fill[y0:y1, x0:x1][new_gray] = np.uint8(np.clip(color, 0, 255))
        sub_hole |= strokes
        gray_text[y0:y1, x0:x1] |= new_gray

    return hole, fill, gray_text


def _tint_ink_mask(
    crop_gray: np.ndarray,
    knockout_text_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean ink mask using a per-pixel local background threshold.

    ``ink_T(y, x) = local_bg(y, x) - _TINT_TEXT_DELTA`` tracks the background
    tone at each pixel instead of one value for the whole zone.  Bright
    knockout text/paper is excluded from the local background estimator so it
    cannot raise the threshold and turn its own scan grain into black ink.
    OCR boxes classified as knockout text are removed from the ink mask
    altogether: they are text semantically, but not black bilevel strokes
    visually.  Weak ink pixels are kept only if they connect to a strong core.
    """
    local_bg = _local_tint_background(crop_gray)
    gray_f = crop_gray.astype(np.float32)

    weak_t = np.maximum(_TINT_LO, local_bg - float(_TINT_TEXT_DELTA))
    strong_t = np.maximum(_TINT_LO, local_bg - float(_TINT_STRONG_TEXT_DELTA))
    weak = gray_f < weak_t
    strong = (gray_f < strong_t) | (crop_gray <= _TINT_STRONG_ABS_MAX)

    # Hysteresis: weak pixels are accepted only when their connected component
    # contains at least one strong pixel.  This is intentionally component-
    # based rather than another morphology pass so thin real strokes survive.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        weak.astype(np.uint8), connectivity=8
    )
    ink = np.zeros_like(weak, dtype=np.uint8)
    if n_labels > 1:
        strong_labels = np.unique(labels[strong & weak])
        strong_labels = strong_labels[strong_labels != 0]
        if strong_labels.size:
            ink[np.isin(labels, strong_labels)] = 1

    if knockout_text_mask is not None:
        ink[knockout_text_mask] = 0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > _TINT_EDGE_NOISE_MAX_AREA:
            continue
        component = labels == label
        if int(crop_gray[component].min()) <= _TINT_EDGE_NOISE_DARK_CORE_MAX:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        pad = _TINT_EDGE_NOISE_NEIGHBOR_PX
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(crop_gray.shape[1], x + ww + pad)
        y1 = min(crop_gray.shape[0], y + hh + pad)
        neighbourhood = crop_gray[y0:y1, x0:x1]
        if (
            int(neighbourhood.max()) - int(neighbourhood.min())
            >= _TINT_EDGE_NOISE_NEIGHBOR_RANGE_MIN
        ):
            ink[component] = 0

    knockout = crop_gray >= _KNOCKOUT_WHITE_MIN
    if knockout.any():
        k = 2 * _KNOCKOUT_DILATE_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        knockout_near = cv2.dilate(knockout.astype(np.uint8), kernel) > 0
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            ink.astype(np.uint8), connectivity=8
        )
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area > _KNOCKOUT_NOISE_MAX_AREA:
                continue
            component = labels == label
            if np.any(component & knockout_near):
                ink[component] = 0

    dk = 2 * _INK_DESPECKLE_PX + 1
    despeckle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, despeckle_kernel)
    return ink > 0


def _try_posterized_tint_overlay_pdf(
    overlay_gray: np.ndarray,
    in_shape: np.ndarray,
    ink: np.ndarray,
    text_hole: np.ndarray | None = None,
    text_fill: np.ndarray | None = None,
) -> tuple[bytes, dict] | None:
    """Return a k4+Flate tint-overlay PDF, or None for JPEG fallback.

    ``text_hole`` marks every text stroke (black ink, white knockout, mid-gray)
    so it is kept out of the background k4 clustering; otherwise the strokes and
    their anti-aliased edges seed a stray cluster that speckles the tint.  Hole
    pixels are written from ``text_fill`` (paper 255 for black/white text, the
    estimated stroke gray for mid-gray text) instead of being quantized.
    """
    if overlay_gray.dtype != np.uint8 or overlay_gray.ndim != 2:
        return None
    if overlay_gray.shape != in_shape.shape or overlay_gray.shape != ink.shape:
        return None
    if text_hole is None:
        text_hole = np.zeros_like(in_shape)
    elif text_hole.shape != overlay_gray.shape:
        return None
    if text_fill is None:
        text_fill = np.full_like(overlay_gray, 255)
    elif text_fill.shape != overlay_gray.shape or text_fill.dtype != np.uint8:
        return None

    valid = in_shape & ~ink & ~text_hole & (overlay_gray < 255)
    valid_count = int(valid.sum())
    if valid_count < max(100, _TINT_POSTERIZE_K):
        return None

    smoothed = _smooth_tint_for_posterize(overlay_gray, valid)
    patch_class = _classify_tint_candidate(smoothed, valid)
    if patch_class == "photo":
        return None
    pixels = smoothed[valid]
    if pixels.size < _TINT_POSTERIZE_K:
        return None

    sample = pixels
    if sample.size > _TINT_POSTERIZE_SAMPLE_MAX:
        step = int(np.ceil(sample.size / _TINT_POSTERIZE_SAMPLE_MAX))
        sample = sample[::step]
    data = sample.astype(np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    try:
        _, _, centers = cv2.kmeans(
            data, _TINT_POSTERIZE_K, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
    except Exception:
        return None

    center_vals = centers.flatten().astype(np.float32)
    labels = _nearest_gray_centers(smoothed[valid].astype(np.float32), center_vals)
    source_pixels = overlay_gray[valid]
    representative_vals = center_vals.copy()
    for idx in range(center_vals.size):
        members = source_pixels[labels == idx]
        if members.size:
            representative_vals[idx] = float(np.median(members))
    labels = _nearest_gray_centers(source_pixels.astype(np.float32), representative_vals)
    quant_vals = np.clip(np.round(representative_vals[labels]), 0, 255).astype(np.uint8)

    quantized = np.full_like(overlay_gray, 255)
    quantized[valid] = quant_vals
    quantized[ink & in_shape] = overlay_gray[ink & in_shape]
    # Punch text strokes out of the quantized field: paper white for black/white
    # text (the base layer carries the glyph), the estimated gray for mid-gray
    # text (the overlay carries it).  Done after the field so a hole always wins.
    hole_in = text_hole & in_shape
    if hole_in.any():
        quantized[hole_in] = text_fill[hole_in]

    diff = np.abs(
        overlay_gray[valid].astype(np.int16) - quantized[valid].astype(np.int16)
    )
    mae = float(diff.mean()) if diff.size else 0.0
    p95 = float(np.percentile(diff, 95)) if diff.size else 0.0
    bad_frac = float((diff > _TINT_POSTERIZE_BAD_DELTA).mean()) if diff.size else 0.0
    if (
        mae > _TINT_POSTERIZE_MAE_MAX
        or p95 > _TINT_POSTERIZE_P95_MAX
        or bad_frac > _TINT_POSTERIZE_BAD_FRAC_MAX
    ):
        return None

    try:
        pdf = _encode_gray_flate_page_pdf(quantized)
    except ValueError:
        # pikepdf rejects page sizes outside [3, 14400] PDF units; a degenerate
        # (near-zero or huge) downsampled tint zone hits this -- fall back to
        # the JPEG/img2pdf path like any other posterize-quality failure.
        return None
    return pdf, {
        "codec": "k4_flate",
        "class": patch_class,
        "mae": mae,
        "p95": p95,
        "bad_frac": bad_frac,
        "valid_pixels": valid_count,
        "hole_pixels": int(hole_in.sum()),
        "centers": [int(round(float(c))) for c in sorted(representative_vals)],
    }


def _smooth_tint_for_posterize(overlay_gray: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median-smooth tint background without letting ink/paper dominate."""
    h, w = overlay_gray.shape[:2]
    vals = overlay_gray[valid]
    fill_value = int(round(float(np.median(vals)))) if vals.size else 255
    work_src = np.where(valid, overlay_gray, fill_value).astype(np.uint8)

    radius = (
        _LOCAL_BG_WINDOW_SMALL_PX
        if max(h, w) <= _LOCAL_BG_SMALL_ZONE_DIM
        else _LOCAL_BG_WINDOW_PX
    )
    max_dim = max(h, w)
    if max_dim > _LOCAL_BG_MAX_DIM:
        scale = _LOCAL_BG_MAX_DIM / float(max_dim)
        work = cv2.resize(
            work_src,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        radius = max(1, int(round(radius * scale)))
    else:
        work = work_src

    k = max(3, 2 * radius + 1)
    if k % 2 == 0:
        k += 1
    smoothed = cv2.medianBlur(work, k)
    if smoothed.shape != overlay_gray.shape:
        smoothed = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_LINEAR)
    return smoothed.astype(np.uint8)


def _classify_tint_candidate(smoothed_gray: np.ndarray, valid: np.ndarray) -> str:
    """Classify tint content using classify_patch on local valid tiles."""
    from .region_class import classify_patch

    fill_value = int(round(float(np.median(smoothed_gray[valid])))) if valid.any() else 255
    h, w = smoothed_gray.shape[:2]
    tile = 256
    classes: list[str] = []
    for y0 in range(0, h, tile):
        y1 = min(h, y0 + tile)
        for x0 in range(0, w, tile):
            x1 = min(w, x0 + tile)
            tile_valid = valid[y0:y1, x0:x1]
            if int(tile_valid.sum()) < 512:
                continue
            patch = np.where(
                tile_valid, smoothed_gray[y0:y1, x0:x1], fill_value
            ).astype(np.uint8)
            classes.append(classify_patch(patch))
    if not classes:
        patch = np.where(valid, smoothed_gray, fill_value).astype(np.uint8)
        return classify_patch(patch)
    photo_frac = classes.count("photo") / len(classes)
    if photo_frac >= 0.5:
        return "photo"
    if "solid_fill" in classes:
        return "solid_fill"
    if "line_art" in classes:
        return "line_art"
    if "text" in classes:
        return "text"
    return classes[0]


def _nearest_gray_centers(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    best_dist = np.full(values.shape, np.inf, dtype=np.float32)
    labels = np.zeros(values.shape, dtype=np.int16)
    for idx, center in enumerate(centers):
        dist = np.abs(values - center)
        take = dist < best_dist
        labels[take] = idx
        best_dist[take] = dist[take]
    return labels


def _encode_gray_flate_page_pdf(gray: np.ndarray) -> bytes:
    """Single-page PDF containing an 8-bit DeviceGray FlateDecode image."""
    import pikepdf

    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("gray must be an HxW uint8 array")
    h, w = gray.shape
    pdf = pikepdf.new()
    image = pikepdf.Stream(
        pdf,
        zlib.compress(np.ascontiguousarray(gray).tobytes()),
        Filter=pikepdf.Name("/FlateDecode"),
    )
    image.Type = pikepdf.Name("/XObject")
    image.Subtype = pikepdf.Name("/Image")
    image.Width = w
    image.Height = h
    image.ColorSpace = pikepdf.Name("/DeviceGray")
    image.BitsPerComponent = 8
    content = pikepdf.Stream(pdf, f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q".encode())
    page = pdf.add_blank_page(page_size=(w, h))
    page.obj.Resources = pikepdf.Dictionary(
        XObject=pikepdf.Dictionary(Im0=pdf.make_indirect(image))
    )
    page.obj.Contents = pdf.make_indirect(content)
    buf = BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def _add_overlay_named(dest, overlay_page, rect, name: str) -> None:
    import pikepdf

    page = pikepdf.Page(dest)
    form = pikepdf.Page(overlay_page).as_form_xobject()
    placed = page.add_resource(form, pikepdf.Name.XObject, name=pikepdf.Name(name))
    if rect is None:
        rect = pikepdf.Rectangle(page.trimbox)
    content = page.calc_form_xobject_placement(
        form, placed, rect, allow_shrink=True, allow_expand=True
    )
    page.contents_add(b"q\n", prepend=True)
    page.contents_add(b"Q\n", prepend=False)
    page.contents_add(content, prepend=False)
    page.contents_coalesce()


def _add_tint_overlays(
    pdf, page, tint_ovs: list, page_idx: int, sources: list
) -> None:
    """Place greyscale tint overlays with /BM /Multiply blend mode.

    Each overlay is a greyscale JPEG image stored in a single-page PDF.
    Multiply blend: overlay_value × bilevel_white = overlay_value (tint shows),
    overlay_value × bilevel_black = 0 (text strokes stay black regardless).
    Paper pixels are 255 in the overlay, so Multiply(1.0, bilevel) = bilevel
    (no change on paper).

    When a contour is present the overlay is clipped to the exact tint-shape
    polygon via a PDF clip path (W/W* n operator) before the XObject is
    painted.  This eliminates the black fringe that occurs when the bounding
    rect extends beyond the actual tint region (e.g. a circular badge or a
    rounded corner), and -- when the zone has holes (a header/column/footer
    frame enclosing the body text) -- keeps the enclosed cavity unpainted.
    """
    import pikepdf

    from .tint_zone import circle_to_pdf_path, contour_to_pdf_path

    resources = page.Resources
    if "/ExtGState" not in resources:
        resources.ExtGState = pikepdf.Dictionary()
    resources.ExtGState[pikepdf.Name("/HPVecFillMultiply")] = pikepdf.Dictionary({
        "/Type": pikepdf.Name("/ExtGState"),
        "/BM": pikepdf.Name("/Multiply"),
    })

    p = pikepdf.Page(page)
    for i, ov in enumerate(tint_ovs):
        ovpdf = pikepdf.open(BytesIO(ov["pdf"]))
        sources.append(ovpdf)
        form = pikepdf.Page(ovpdf.pages[0]).as_form_xobject()
        name = pikepdf.Name(f"/HPTintOv{page_idx}_{i}")
        placed = p.add_resource(form, pikepdf.Name.XObject, name=name)
        rect = pikepdf.Rectangle(*ov["rect"])
        inner = p.calc_form_xobject_placement(
            form, placed, rect, allow_shrink=True, allow_expand=True
        )
        # Nested q/Q: outer activates Multiply; clip path (if any) restricts
        # painting to the actual tint polygon so corners/badges don't overflow.
        # Circles use 4-arc Bézier for mathematical precision; polygons use
        # approxPolyDP-simplified m/l/h paths.
        # Holes (e.g. the body-text cavity enclosed by a header/column/footer
        # frame) are appended as extra subpaths in the same clip path, and
        # the clip operator switches to "W*" (even-odd) so they're excluded
        # from the painted region -- the standard PDF "donut clip" technique.
        if ov.get("circle_fit") is not None:
            cx, cy, r = ov["circle_fit"]
            clip = circle_to_pdf_path(cx, cy, r, ov["page_h_px"])
            clip_op = b"W n\n"
        elif ov.get("contour") is not None:
            holes = ov.get("holes") or []
            clip = contour_to_pdf_path(ov["contour"], ov["page_h_px"], holes)
            clip_op = b"W* n\n" if holes else b"W n\n"
        else:
            clip = b""
            clip_op = b"W n\n"
        if clip:
            content = b"q /HPVecFillMultiply gs\n" + clip + clip_op + inner + b"\nQ\n"
        else:
            content = b"q /HPVecFillMultiply gs\n" + inner + b"\nQ\n"
        p.contents_add(pikepdf.Stream(pdf, content), prepend=False)
        p.contents_coalesce()


def _add_vector_fills(pdf, page, vector_fills: list, page_h_px: int) -> None:
    import pikepdf

    from .vecfill import fill_rect_ops, px_to_pt

    resources = page.Resources
    if "/ExtGState" not in resources:
        resources.ExtGState = pikepdf.Dictionary()
    resources.ExtGState[pikepdf.Name("/HPVecFillMultiply")] = pikepdf.Dictionary({
        "/Type": pikepdf.Name("/ExtGState"),
        "/BM": pikepdf.Name("/Multiply"),
    })

    scale = px_to_pt(1, 72)
    parts = [b"q /HPVecFillMultiply gs\n"]
    for fill in vector_fills:
        x0, y0, x1, y1 = fill["rect"]
        parts.append(fill_rect_ops(x0, y0, x1, y1, fill["gray"], page_h_px, scale))
    parts.append(b"Q\n")
    pikepdf.Page(page).contents_add(pikepdf.Stream(pdf, b"".join(parts)))


def _add_vector_tint_fills(pdf, page, vector_tint_fills: list, page_h_px: int) -> None:
    import pikepdf
    from .tint_zone import contour_to_pdf_path

    resources = page.Resources
    if "/ExtGState" not in resources:
        resources.ExtGState = pikepdf.Dictionary()
    resources.ExtGState[pikepdf.Name("/HPVecFillMultiply")] = pikepdf.Dictionary({
        "/Type": pikepdf.Name("/ExtGState"),
        "/BM": pikepdf.Name("/Multiply"),
    })

    parts = [b"q /HPVecFillMultiply gs\n"]
    for fill in vector_tint_fills:
        gray = fill["gray"]
        contour = fill["contour"]
        holes = fill["holes"]

        color_op = f"{gray:.4f} g\n".encode('ascii')
        path_bytes = contour_to_pdf_path(contour, page_h_px, holes)
        fill_op = b"f*\n" if holes else b"f\n"

        parts.append(color_op + path_bytes + fill_op)
    parts.append(b"Q\n")

    pikepdf.Page(page).contents_add(pikepdf.Stream(pdf, b"".join(parts)))
