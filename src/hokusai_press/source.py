"""Page ingest.

Principle (non-destructive): the final image layer must be computed from the
highest-fidelity *original* pixels. For scan PDFs that means the embedded
page image (extracted losslessly via pikepdf), not a rasterization. The
300 dpi render is only the working resolution for OCR/geometry; the scale
factor back to original pixels is recorded so every box maps both ways.

load_page() yields (SourceRef, original_bgr, ocr_bgr) per page:
- original_bgr : full-resolution original (embedded image, or the image file)
- ocr_bgr      : downscaled to ~OCR_DPI for detection/recognition
- SourceRef.ocr_scale : ocr_px = original_px * ocr_scale
"""

from __future__ import annotations

import os
import threading
from typing import Iterator

import cv2
import numpy as np

from .model import SourceRef

OCR_DPI = 300
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".jp2", ".webp")

# pdfium (pypdfium2) is NOT thread-safe: concurrent document/page access from
# the review UI's threadpool corrupts native state and crashes the process
# ("Data format error" / "Failed to load page", then a hard exit). Serialize
# every pdfium call through one process-wide lock. Pure-numpy work (downscale,
# color convert) stays outside the lock so it still parallelizes.
_PDFIUM_LOCK = threading.RLock()


def _imread_unicode(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return img


def _read_image_dpi(path: str) -> float | None:
    """Return image DPI from PIL metadata if present and valid, else None."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            dpi_info = im.info.get("dpi")
            if isinstance(dpi_info, (tuple, list)) and len(dpi_info) >= 2:
                dx, dy = float(dpi_info[0]), float(dpi_info[1])
                if dx > 1.0 and dy > 1.0:
                    return (dx + dy) / 2.0
            elif isinstance(dpi_info, (int, float)) and float(dpi_info) > 1.0:
                return float(dpi_info)
    except Exception:
        pass
    return None


def _downscale_for_ocr(img: np.ndarray, src_dpi: float | None) -> tuple[np.ndarray, float]:
    """Return (ocr_image, ocr_scale). If src dpi unknown, only downscale very
    large images so OCR stays fast; ocr_scale records the mapping."""
    h, w = img.shape[:2]
    if src_dpi and src_dpi > OCR_DPI:
        scale = OCR_DPI / src_dpi
    elif max(h, w) > 3000:
        scale = 3000 / max(h, w)
    else:
        scale = 1.0
    if scale >= 0.999:
        return img, 1.0
    ocr = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                     interpolation=cv2.INTER_AREA)
    return ocr, scale


def pdf_page_sizes(path: str) -> list[tuple[float, float]]:
    """Return (width_pt, height_pt) for each page of a PDF without decoding or
    rendering pixels (via pypdfium2 page.get_size()). Returns [] for non-PDFs
    or missing files."""
    ext = os.path.splitext(path)[1].lower()
    if ext != ".pdf" or not os.path.exists(path):
        return []
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        doc = pdfium.PdfDocument(path)
        try:
            return [doc[i].get_size() for i in range(len(doc))]
        finally:
            doc.close()


def load_page(
    path: str, pages: "set[int] | None" = None
) -> Iterator[tuple[SourceRef, np.ndarray, np.ndarray, float | None]]:
    """Yield (SourceRef, original_bgr, ocr_bgr, dpi) for each page.

    If `pages` is given (a set of 0-based page indices), only those pages are
    rasterized and yielded -- so debugging a few pages is fast. The yielded
    SourceRef.page_index keeps the TRUE pdf index, so re-extraction and the
    review UI still line up.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext != ".pdf":
        if pages is not None and 0 not in pages:
            return
        img = _imread_unicode(path)
        dpi = _read_image_dpi(path)
        ocr, scale = _downscale_for_ocr(img, dpi)
        yield SourceRef(path=path, ocr_scale=scale), img, ocr, dpi
        return

    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        doc = pdfium.PdfDocument(path)
        n = len(doc)
    try:
        for i in range(n):
            if pages is not None and i not in pages:
                continue
            # extract under the lock; downscale (pure numpy) outside it
            with _PDFIUM_LOCK:
                original, dpi, img_id = _extract_original(doc[i], doc, i)
            ocr, scale = _downscale_for_ocr(original, dpi)
            yield (
                SourceRef(path=path, page_index=i, embedded_image_id=img_id,
                          ocr_scale=scale),
                original,
                ocr,
                dpi,
            )
    finally:
        with _PDFIUM_LOCK:
            doc.close()


def load_single(path: str, page_index: int) -> tuple[np.ndarray, float | None]:
    """Re-extract one page's original image (for the review UI / re-render)."""
    if not os.path.exists(path):
        raise ValueError(f"source file not found: '{path}'")
    ext = os.path.splitext(path)[1].lower()
    if ext != ".pdf":
        if page_index != 0:
            raise ValueError(
                f"cannot read page {page_index + 1} (index {page_index}) from "
                f"single-page image '{path}'"
            )
        return _imread_unicode(path), _read_image_dpi(path)
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        try:
            doc = pdfium.PdfDocument(path)
        except Exception as e:
            raise ValueError(f"cannot open source PDF '{path}': {e}") from e
        try:
            n = len(doc)
            if page_index < 0 or page_index >= n:
                raise ValueError(
                    f"source PDF '{path}' has only {n} page(s), but DB expects "
                    f"page {page_index + 1} (index {page_index}); re-run "
                    f"`hokusai-press run` on '{path}' to update the DB"
                )
            original, dpi, _ = _extract_original(doc[page_index], doc, page_index)
        finally:
            doc.close()
    return original, dpi


_ROTATE_CV2 = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# Tolerance for a full-page embedded image covering the page's visible box
# (CropBox intersected with MediaBox): each edge within 2% of page width/height,
# and off-diagonal CTM terms within 1% of the dominant axis scale.
_FULL_PAGE_EDGE_TOL = 0.02
_CTM_SKEW_TOL = 0.01


def _simple_image_ctm_rotation(
    a: float, b: float, c: float, d: float
) -> int | None:
    """Return clockwise rotation (0, 90, 180, 270) for a pure scale + 90-deg-
    multiple rotation CTM (no flip, no significant skew), or None."""
    # Positive determinant means orientation-preserving (rotation + positive
    # scale, not a reflection/flip).
    det = a * d - b * c
    scale = max(abs(a), abs(b), abs(c), abs(d))
    if scale <= 0 or det <= 0:
        return None
    if max(abs(b), abs(c)) <= _CTM_SKEW_TOL * min(abs(a), abs(d)):
        if a > 0 and d > 0:
            return 0
        if a < 0 and d < 0:
            return 180
        return None
    if max(abs(a), abs(d)) <= _CTM_SKEW_TOL * min(abs(b), abs(c)):
        if b < 0 and c > 0:
            return 90
        if b > 0 and c < 0:
            return 270
        return None
    return None


def _can_extract_single_image(page) -> tuple[object, int] | None:
    """Return (img_obj, ctm_rotation_deg) when the page consists solely of one
    full-page embedded image (plus optional invisible text-render-mode-3 OCR
    glyphs and container Form XObjects), else None."""
    import pypdfium2.raw as pdfium_c

    images = []
    for obj in page.get_objects(max_depth=16):
        t = obj.type
        if t == pdfium_c.FPDF_PAGEOBJ_IMAGE:
            images.append(obj)
            if len(images) > 1:
                return None
        elif t == pdfium_c.FPDF_PAGEOBJ_FORM:
            continue
        elif t == pdfium_c.FPDF_PAGEOBJ_TEXT:
            if (
                pdfium_c.FPDFTextObj_GetTextRenderMode(obj)
                != pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE
            ):
                return None
        else:
            # FPDF_PAGEOBJ_PATH, FPDF_PAGEOBJ_SHADING, etc.
            return None

    if len(images) != 1:
        return None

    img_obj = images[0]
    cb_l, cb_b, cb_r, cb_t = page.get_bbox()
    cw, ch = cb_r - cb_l, cb_t - cb_b
    if cw <= 0 or ch <= 0:
        return None

    il, ib, ir, it = img_obj.get_pos()
    if (
        abs(il - cb_l) > _FULL_PAGE_EDGE_TOL * cw
        or abs(ir - cb_r) > _FULL_PAGE_EDGE_TOL * cw
        or abs(ib - cb_b) > _FULL_PAGE_EDGE_TOL * ch
        or abs(it - cb_t) > _FULL_PAGE_EDGE_TOL * ch
    ):
        return None

    a, b, c, d, _, _ = img_obj.get_matrix().get()
    ctm_rot = _simple_image_ctm_rotation(a, b, c, d)
    if ctm_rot is None:
        return None

    return img_obj, ctm_rot


def _extract_original(page, doc, index: int) -> tuple[np.ndarray, float | None, str | None]:
    """Prefer the single embedded full-page image (lossless original). Fall
    back to a 300 dpi rasterization when a page is not a simple scan."""
    # get_size() already reflects /Rotate (it's the page's DISPLAY size), but
    # the embedded image bytes are in the page's own unrotated frame -- e.g. a
    # wide table scanned sideways and tagged /Rotate=270 so it displays as a
    # tall page (the reader turns the BOOK, not the table). Apply that same
    # rotation to the extracted bitmap (a lossless 90 deg permutation, not a
    # resample) so the original matches what every viewer/the rest of the
    # pipeline treats as "this page", instead of staying landscape while
    # every other page in the book is portrait.
    page_w_pt, _ = page.get_size()
    rotation = page.get_rotation() % 360

    cand = _can_extract_single_image(page)
    if cand is not None:
        img_obj, ctm_rot = cand
        try:
            pil = img_obj.get_bitmap(render=False).to_pil()
            arr = np.array(pil.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            total_rot = (rotation + ctm_rot) % 360
            cv2_rot = _ROTATE_CV2.get(total_rot)
            if cv2_rot is not None:
                bgr = cv2.rotate(bgr, cv2_rot)
            # dpi from embedded pixel size vs page point size (both now in
            # the same, rotation-applied orientation)
            dpi = bgr.shape[1] / (page_w_pt / 72.0) if page_w_pt else None
            return bgr, dpi, "img0"
        except Exception:
            pass

    # pypdfium2's default render output is BGR(A/X) when rev_byteorder=False.
    bitmap = page.render(scale=OCR_DPI / 72, rev_byteorder=False)
    bgr = np.ascontiguousarray(bitmap.to_numpy()[..., :3])
    return bgr, float(OCR_DPI), None


# --- paper and orientation ----------------------------------------------------
# A flatbed/ADF scan whose page-size detection failed (a black chapter page is
# the usual case) is saved larger than the sheet, with the uniform gray
# scanner bed around it. Measured on img20261121_0010/0013: bed ~123 gray with
# a column std of ~1, page 67-80 (black) or ~250 (white), plus a pure-white
# 31px pad at the image border.
BED_TOL = 10           # a bed column/row is within this of the bed gray ...
BED_STD = 4.0          # ... and this uniform
BED_MIN_FRAC = 0.04    # a bed must be at least this wide to count
PAPER_MAX_FRAC = 0.92  # ... and leave the sheet smaller than this
PAPER_MIN_FRAC = 0.30  # a "sheet" narrower than this is a gray band on the page


def _white_pad(gray: np.ndarray, axis: int) -> tuple[int, int]:
    """[lo, hi) along an axis inside the pure-white border padding: the run of
    flat 255 lines contiguous with each edge (white paper is never this flat),
    plus the few pixels over which the pad's inner edge fades."""
    g = gray.astype(np.float32)
    flat = (np.median(g, axis=axis) >= 254) & (g.std(axis=axis) < 2.0)
    n = flat.size
    lo = 0
    while lo < n and flat[lo]:
        lo += 1
    hi = n
    while hi > lo and flat[hi - 1]:
        hi -= 1
    if lo == 0 and hi == n:
        return 0, n
    fade = 4
    return min(n, lo + fade if lo else 0), max(0, hi - fade if hi < n else n)


def _bed_span(gray: np.ndarray, axis: int, bed_gray: "float | None" = None):
    """[a, b) of the sheet along one axis of an unpadded image (axis 0 =
    columns), or None if there is no uniform bed on both sides."""
    g = gray.astype(np.float32)
    mean = g.mean(axis=axis)
    std = g.std(axis=axis)
    n = mean.size
    k = max(3, int(0.01 * n))
    bed_l = float(np.median(mean[k:2 * k]))
    bed_r = float(np.median(mean[-2 * k:-k]))
    if abs(bed_l - bed_r) > BED_TOL or not (20 < bed_l < 235):
        return None                               # no gray bed on both sides
    if bed_gray is not None and abs((bed_l + bed_r) / 2 - bed_gray) > BED_TOL:
        return None                               # not the same scanner bed
    # uniform relative to the bed's own noise (1-4 gray levels measured)
    std_lim = max(BED_STD, 2.0 * float(np.median(np.r_[std[k:2 * k], std[-2 * k:-k]])))
    bed = (np.abs(mean - (bed_l + bed_r) / 2) <= BED_TOL) & (std <= std_lim)
    dark = np.flatnonzero(mean < 240)           # skip what is left of the pad
    if dark.size == 0:
        return None
    lo, hi = int(dark[0]), int(dark[-1]) + 1
    a = lo
    while a < hi and bed[a]:
        a += 1
    b = hi
    while b > a and bed[b - 1]:
        b -= 1
    if (a - lo) < BED_MIN_FRAC * n or (hi - b) < BED_MIN_FRAC * n:
        return None
    if not (PAPER_MIN_FRAC * n <= (b - a) <= PAPER_MAX_FRAC * n):
        return None
    return a, b, (bed_l + bed_r) / 2


BED_SEED_MIN = 0.10    # bed gray must cover this much of the ring inside the pad ...
BED_MAX_FRAC = 0.50    # ... and the bed less than this much of the image
BED_MIN_AREA = 0.002   # a bed piece is at least this much of the image
BED_NOISE_MAX = 2.0    # ... and flat: median 7x7 std (bed 0.3-0.8; print >= 4)
PAD_MAX_FRAC = 0.03    # the scanner's white pad: a thin band on all four sides
BED_EDGE_GROW = 0.01   # the sheet's edge shadow: this far (of the short side) from the bed
BED_EDGE_BAND = 0.04   # ... and any distance within this outer band of the image
BED_EDGE_PAPER = 200   # ... over pixels darker than paper


def scanner_bed_mask(gray: np.ndarray) -> "np.ndarray | None":
    """Pixels outside the sheet -- the white border pad and the uniform gray
    scanner bed -- as a bool mask, or None when the image shows no bed.

    A sheet laid askew leaves bed triangles at its corners (img20261121_0010
    p243-244); unlike find_paper, which wants whole bed columns, this follows
    the bed itself: gray within BED_TOL of the bed's own level, connected to
    the border of the unpadded image, in pieces too large to be ink and as
    flat as only the bed is. Measured on img20261121: the bed's median 7x7
    std is 0.3-0.8; printed gray touching the paper edge (a tinted chapter
    band 0006 p166, a header bar 0014 p36, a cover 0003 p1) and paper
    texture 4-29 -- level and connectivity alone took those for bed."""
    h, w = gray.shape
    px0, px1 = _white_pad(gray, 0)
    py0, py1 = _white_pad(gray, 1)
    # Only inside the scanner's own pad: a page's flat white margin around
    # flat (born-digital) gray print is no pad, and that print no bed.
    for lo, hi, n in ((px0, px1, w), (py0, py1, h)):
        if not (0 < lo <= PAD_MAX_FRAC * n and 0 < n - hi <= PAD_MAX_FRAC * n):
            return None
    ih, iw = py1 - py0, px1 - px0
    k = max(3, int(0.01 * min(ih, iw)))
    if ih <= 2 * k or iw <= 2 * k:
        return None                              # all pad: a blank white page
    inner = cv2.medianBlur(np.ascontiguousarray(gray[py0:py1, px0:px1]), 5)
    ring = np.r_[inner[:k].ravel(), inner[-k:].ravel(),
                 inner[k:-k, :k].ravel(), inner[k:-k, -k:].ravel()]
    mid = ring[(ring > 20) & (ring < 235)]
    if mid.size < BED_SEED_MIN * ring.size:
        return None
    bed_gray = int(np.argmax(np.bincount(mid, minlength=256)))
    if np.count_nonzero(np.abs(ring.astype(np.int16) - bed_gray) <= BED_TOL) < BED_SEED_MIN * ring.size:
        return None
    cand = (np.abs(inner.astype(np.int16) - bed_gray) <= BED_TOL).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    raw = gray[py0:py1, px0:px1].astype(np.float32)
    mu = cv2.blur(raw, (7, 7))
    sd = np.sqrt(np.maximum(cv2.blur(raw * raw, (7, 7)) - mu * mu, 0.0))
    core = cv2.erode(cand, np.ones((9, 9), np.uint8)).astype(bool)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches = x <= k or y <= k or x + cw >= iw - k or y + ch >= ih - k
        if not (touches and area >= BED_MIN_AREA * ih * iw):
            continue
        r = core[y:y + ch, x:x + cw] & (lbl[y:y + ch, x:x + cw] == i)
        keep[i] = r.any() and float(np.median(sd[y:y + ch, x:x + cw][r])) <= BED_NOISE_MAX
    bed = keep[lbl]
    if not bed.any() or bed.mean() >= BED_MAX_FRAC:
        return None
    mask = np.ones((h, w), bool)
    mask[py0:py1, px0:px1] = bed
    # the sheet's edge casts a darker line on the bed (105-117 against 123)
    # and, where it lifts, a dark sliver (75-90) out to the bed's tip: add
    # what is darker than paper next to the bed, and what runs on from it
    # within the outer band of the image -- never the paper, never print
    # inside the page
    dark = gray < BED_EDGE_PAPER
    grown = mask.astype(np.uint8)
    for _ in range(max(3, int(BED_EDGE_GROW * min(h, w)))):   # along the edge
        nxt = cv2.dilate(grown, np.ones((3, 3), np.uint8)) & (dark | mask).astype(np.uint8)
        if np.array_equal(nxt, grown):
            break
        grown = nxt
    mask = grown.astype(bool)
    bx, by = int(BED_EDGE_BAND * w), int(BED_EDGE_BAND * h)
    band = np.ones((h, w), bool)
    band[by:h - by, bx:w - bx] = False
    reach = (dark & band) | mask                                # out to the tips
    n, lbl = cv2.connectedComponents(reach.astype(np.uint8), connectivity=8)
    hit = np.zeros(n, bool)
    hit[np.unique(lbl[mask])] = True
    hit[0] = False
    return hit[lbl] | mask


def find_paper(img: np.ndarray) -> "list | None":
    """The sheet's box [x0, y0, x1, y1] when the image shows a uniform scanner
    bed around it (left+right or top+bottom), else None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    px0, px1 = _white_pad(gray, 0)
    py0, py1 = _white_pad(gray, 1)
    inner = gray[py0:py1, px0:px1]
    if inner.size == 0:
        return None
    # one axis first, then the other inside it, and only if it shows the
    # same bed gray (a dark page's own blank rows are not the scanner bed)
    xs = _bed_span(inner, 0)
    if xs is not None:
        ys = _bed_span(inner[:, xs[0]:xs[1]], 1, bed_gray=xs[2])
    else:
        ys = _bed_span(inner, 1)
        if ys is not None:
            xs = _bed_span(inner[ys[0]:ys[1], :], 0, bed_gray=ys[2])
    if xs is None and ys is None:
        return None
    x0, x1 = (px0 + xs[0], px0 + xs[1]) if xs else (0, w)
    y0, y1 = (py0 + ys[0], py0 + ys[1]) if ys else (0, h)
    return [float(x0), float(y0), float(x1), float(y1)]


def _estimate_bed_gray_1d(inner: np.ndarray, axis: int) -> float | None:
    """Estimate the uniform scanner bed gray level from the outer strips along
    an axis (axis=0 -> left/right strips, axis=1 -> top/bottom strips)."""
    n = inner.shape[1] if axis == 0 else inner.shape[0]
    k = max(3, int(0.02 * n))
    if 2 * k >= n:
        return None
    if axis == 0:
        s_l, s_r = inner[:, :k], inner[:, -k:]
    else:
        s_l, s_r = inner[:k, :], inner[-k:, :]
    med_l, med_r = float(np.median(s_l)), float(np.median(s_r))
    std_l, std_r = float(np.std(s_l)), float(np.std(s_r))
    valid_l = 20.0 < med_l < 235.0
    valid_r = 20.0 < med_r < 235.0
    if valid_l and valid_r and abs(med_l - med_r) <= 25.0:
        return (med_l + med_r) / 2.0
    if valid_l and std_l < 8.0:
        return med_l
    if valid_r and std_r < 8.0:
        return med_r
    return None


def _find_window_1d(inner: np.ndarray, axis: int, win_len: int) -> tuple[int, int] | None:
    """Slide a window of width `win_len` along `axis` (0=x, 1=y) to locate the
    sheet that differs most from the scanner bed, then snap edges."""
    n = inner.shape[1] if axis == 0 else inner.shape[0]
    if win_len <= 0 or win_len >= int(0.92 * n):
        return None
    bed_gray = _estimate_bed_gray_1d(inner, axis)
    if bed_gray is None:
        return None

    g = inner.astype(np.float32)
    diff = np.abs(g - bed_gray).mean(axis=axis)
    sd = g.std(axis=axis)
    score = diff + sd

    cs = np.concatenate([[0.0], np.cumsum(score, dtype=np.float64)])
    win_scores = cs[win_len:] - cs[:-win_len]
    bx0 = int(np.argmax(win_scores))
    bx1 = bx0 + win_len

    in_score = float(np.mean(score[bx0:bx1]))
    out_parts = []
    if bx0 > 0:
        out_parts.append(score[:bx0])
    if bx1 < n:
        out_parts.append(score[bx1:])
    if not out_parts:
        return None
    out_score = float(np.median(np.concatenate(out_parts)))
    if in_score - out_score < 8.0 and in_score < 2.0 * max(out_score, 1.0):
        return None

    # Snap left/right edges to where the page stands out above the bed
    edge_sig = diff + 0.5 * sd
    in_edge = float(np.median(edge_sig[bx0:bx1]))
    thr = max(8.0, 0.45 * out_score + 0.35 * in_edge)
    pad = max(10, int(0.08 * win_len))

    lo_lim = max(0, bx0 - pad)
    hi_lim = min(n, bx1 + pad)
    a = bx0
    while a > lo_lim and edge_sig[a - 1] >= thr:
        a -= 1
    while a < bx1 - 1 and edge_sig[a] < thr:
        a += 1
    b = bx1
    while b < hi_lim and edge_sig[b] >= thr:
        b += 1
    while b > a + 1 and edge_sig[b - 1] < thr:
        b -= 1
    if b - a < int(0.70 * win_len):
        return bx0, bx1
    return a, b


def find_paper_by_book_size(
    img: np.ndarray, target_w: float, target_h: float,
) -> "list[float] | None":
    """Locate the sheet [x0, y0, x1, y1] in an oversized scan using the book's
    normal page size (target_w, target_h in this image's pixels) as a guide."""
    if target_w <= 0 or target_h <= 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    px0, px1 = _white_pad(gray, 0)
    py0, py1 = _white_pad(gray, 1)
    inner = gray[py0:py1, px0:px1]
    ih, iw = inner.shape
    if ih == 0 or iw == 0:
        return None

    big_x = iw > target_w * 1.25
    big_y = ih > target_h * 1.25
    if big_x and not big_y:
        win_w = max(1, min(iw - 1, int(round(ih * (target_w / target_h)))))
        xs = _find_window_1d(inner, 0, win_w)
        if xs is None:
            return None
        return [float(px0 + xs[0]), 0.0, float(px0 + xs[1]), float(h)]
    if big_y and not big_x:
        win_h = max(1, min(ih - 1, int(round(iw * (target_h / target_w)))))
        ys = _find_window_1d(inner, 1, win_h)
        if ys is None:
            return None
        return [0.0, float(py0 + ys[0]), float(w), float(py0 + ys[1])]
    if big_x and big_y:
        win_w = max(1, min(iw - 1, int(round(target_w))))
        xs = _find_window_1d(inner, 0, win_w)
        if xs is None:
            return None
        win_h = max(1, min(ih - 1, int(round(target_h))))
        ys = _find_window_1d(inner[:, xs[0]:xs[1]], 1, win_h)
        if ys is None:
            return [float(px0 + xs[0]), 0.0, float(px0 + xs[1]), float(h)]
        return [float(px0 + xs[0]), float(py0 + ys[0]),
                float(px0 + xs[1]), float(py0 + ys[1])]
    return None


def orient(img: np.ndarray, ref: SourceRef, scale: float = 1.0) -> np.ndarray:
    """Apply ref.paper (crop) and ref.rotate (clockwise) to a raw page image;
    scale maps raw pixels to img's pixels (ref.ocr_scale for the OCR raster)."""
    if ref.paper:
        x0, y0, x1, y1 = (int(round(v * scale)) for v in ref.paper)
        img = img[max(0, y0):y1, max(0, x0):x1]
    r = int(ref.rotate or 0) % 360
    if r:
        img = cv2.rotate(img, _ROTATE_CV2[r])
    return np.ascontiguousarray(img)


def load_oriented(ref: SourceRef) -> np.ndarray:
    """The page image as analyzed: raw extraction + orient()."""
    original, _ = load_single(ref.path, ref.page_index)
    return orient(original, ref)
