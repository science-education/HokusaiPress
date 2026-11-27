"""Fetch book covers (openBD / hanmoto.com / Google Books), upscale without
adding margins, and prepend cover + blank verso page to a PDF.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import naming

logger = logging.getLogger(__name__)

USER_AGENT = naming.USER_AGENT
OPENBD_ENDPOINT = "https://api.openbd.jp/v1/get"
HANMOTO_BOOK_URL = "https://www.hanmoto.com/bd/isbn/{isbn}"
GOOGLE_BOOKS_ENDPOINT = "https://www.googleapis.com/books/v1/volumes"

DEFAULT_SOURCES = ("openbd", "hanmoto", "google")
DEFAULT_MIN_LONG_EDGE = 250
DEFAULT_TARGET_DPI = 300
DEFAULT_JPEG_QUALITY = 94
COVER_META_KEY = "/HokusaiPressCover"


@dataclass
class CoverResult:
    isbn: str
    image_bgr: np.ndarray
    source: str
    url: str
    cached: bool = False


@dataclass
class CoverInsertResult:
    src_pdf: str
    dst_pdf: str
    isbn: str | None
    source: str
    cover_size_px: tuple[int, int]
    upscale_method: str
    cover_box_pt: tuple[float, float]
    blank_back: bool
    skipped: bool = False
    note: str = ""


# --- ISBN helpers ---------------------------------------------------------------

def isbn10_to_13(isbn: str) -> str:
    """Normalize ISBN-10 or ISBN-13 to a valid 13-digit ISBN string."""
    digits = re.sub(r"[^0-9Xx]", "", isbn).upper()
    if len(digits) == 13 and naming.isbn_ok(digits):
        return digits
    if len(digits) == 10 and naming.isbn_ok(digits):
        core = "978" + digits[:9]
        total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(core))
        check = (10 - total % 10) % 10
        return core + str(check)
    raise ValueError(f"invalid ISBN: {isbn}")


def _db_page_texts(db_path: str, pdf_path: str, first: int = 8, last: int = 15) -> tuple[dict[int, str], str]:
    """Read stored OCR text from `db_path` when the target PDF has no text layer."""
    from .store import Store

    if not os.path.exists(db_path):
        return {}, ""

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    candidates = [os.path.basename(pdf_path)]
    # Support outputs with suffixes like `_HP` or `（OCR済）` by matching prefix
    store = Store(db_path)
    try:
        doc_ids = store.doc_ids()
        matched_id = None
        for cand in candidates:
            if cand in doc_ids:
                matched_id = cand
                break
        if matched_id is None:
            for did in doc_ids:
                dstem = os.path.splitext(did)[0]
                if dstem and stem.startswith(dstem):
                    matched_id = did
                    break
        if matched_id is None:
            return {}, ""
        rows = store.list_pages(matched_id)
    finally:
        store.close()

    if not rows:
        return {}, ""
    n = len(rows)
    tail: dict[int, str] = {}
    for row in rows[max(0, n - last):]:
        tail[row.page_index + 1] = row.params.reading_text()
    head_parts = []
    for row in rows[:min(first, n)]:
        txt = row.params.reading_text()
        head_parts.append(txt)
        tail.setdefault(row.page_index + 1, txt)
    return tail, "\n".join(head_parts)


def _ocr_tail_texts(pdf_path: str, engine, first: int = 3, last: int = 10, dpi: int = 200) -> tuple[dict[int, str], str]:
    """OCR only opening and closing pages to locate an ISBN on image-only PDFs."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    try:
        n = len(doc)
        indices = sorted(set(range(min(first, n))) | set(range(max(0, n - last), n)))
        texts: dict[int, str] = {}
        for idx in indices:
            page = doc[idx]
            bmp = page.render(scale=dpi / 72.0)
            rgb = np.asarray(bmp.to_pil().convert("RGB"))
            bmp.close()
            page.close()
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            res = engine(bgr)
            texts[idx + 1] = "\n".join((ln.get("text") or "") for ln in res.get("lines", []))
        head = "\n".join(texts.get(i + 1, "") for i in range(min(first, n)))
        return texts, head
    finally:
        doc.close()


def extract_isbn(
    pdf_path: str,
    *,
    db_path: str | None = None,
    ocr_engine=None,
    lookup=naming.lookup_isbn,
) -> tuple[str | None, str]:
    """Find the most likely 13-digit ISBN for *pdf_path*.

    Checks in order: (1) embedded PDF text layer, (2) SQLite Store OCR text,
    (3) lightweight tail/head OCR if *ocr_engine* is supplied.
    Returns ``(isbn13_or_none, origin_note)``.
    """
    pages, head = naming.page_texts(pdf_path)
    origin = "pdf_text"

    if not any(t.strip() for t in pages.values()):
        if db_path:
            pages, head = _db_page_texts(db_path, pdf_path)
            origin = "db"
        if not any(t.strip() for t in pages.values()) and ocr_engine is not None:
            pages, head = _ocr_tail_texts(pdf_path, ocr_engine)
            origin = "ocr_fallback"

    isbn_pages: dict[str, list[int]] = {}
    colophon: set[str] = set()
    for no in sorted(pages, reverse=True):
        t = pages[no]
        for raw_isbn in naming.find_isbns(t):
            try:
                isbn13 = isbn10_to_13(raw_isbn)
            except ValueError:
                continue
            isbn_pages.setdefault(isbn13, []).append(no)
            if any(w in t for w in naming.COLOPHON_WORDS):
                colophon.add(isbn13)

    if not isbn_pages:
        return None, "ISBNが見つからない"

    candidates = list(isbn_pages.keys())
    if len(candidates) == 1:
        return candidates[0], origin

    # Disambiguate when back-cover ads list multiple ISBNs.
    best_score = -1.0
    best_isbn = candidates[0]
    for isbn13 in candidates:
        score = 0.3 if isbn13 in colophon else 0.0
        if lookup is not None:
            try:
                rec = lookup(isbn13)
                if rec is not None:
                    score += naming.title_score(rec.title, head)
            except Exception as e:
                logger.debug("ISBN lookup failed for %s: %s", isbn13, e)
        if score > best_score:
            best_score = score
            best_isbn = isbn13
    return best_isbn, origin


# --- Cover fetching -------------------------------------------------------------

def _http_get(url: str, opener=None, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with (opener or urllib.request.urlopen)(req, timeout=timeout) as resp:
        return resp.read()


def _decode_image(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return img


def _openbd_cover_url(isbn13: str, opener=None) -> str | None:
    url = f"{OPENBD_ENDPOINT}?{urllib.parse.urlencode({'isbn': isbn13})}"
    payload = json.loads(_http_get(url, opener=opener).decode("utf-8"))
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return None
    cover = (payload[0].get("summary") or {}).get("cover") or ""
    return cover.strip() or None


_OG_IMAGE_RE = re.compile(
    r'<meta\s+[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_OG_IMAGE_REV_RE = re.compile(
    r'<meta\s+[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:image["\']',
    re.IGNORECASE,
)
_HANMOTO_IMG_RE = re.compile(
    r'(https?://www\.hanmoto\.com/bd/img/[^"\'\s<>]+|/bd/img/[^"\'\s<>]+)',
    re.IGNORECASE,
)


def _hanmoto_cover_url(isbn13: str, opener=None) -> str | None:
    page_url = HANMOTO_BOOK_URL.format(isbn=urllib.parse.quote(isbn13))
    body = _http_get(page_url, opener=opener).decode("utf-8", errors="replace")
    for pattern in (_OG_IMAGE_RE, _OG_IMAGE_REV_RE):
        m = pattern.search(body)
        if m:
            found = html.unescape(m.group(1)).strip()
            if found and "noimage" not in found.lower():
                return urllib.parse.urljoin(page_url, found)
    m = _HANMOTO_IMG_RE.search(body)
    if m:
        found = html.unescape(m.group(1)).strip()
        if "noimage" not in found.lower():
            return urllib.parse.urljoin(page_url, found)
    return None


def _google_books_cover_url(isbn13: str, opener=None) -> str | None:
    url = f"{GOOGLE_BOOKS_ENDPOINT}?{urllib.parse.urlencode({'q': f'isbn:{isbn13}'})}"
    payload = json.loads(_http_get(url, opener=opener).decode("utf-8"))
    items = payload.get("items") or []
    if not items:
        return None
    links = (items[0].get("volumeInfo") or {}).get("imageLinks") or {}
    for key in ("extraLarge", "large", "medium", "small", "thumbnail", "smallThumbnail"):
        val = links.get(key)
        if val:
            val = re.sub(r"^http://", "https://", val.strip())
            return val.replace("&edge=curl", "")
    return None


_SOURCE_RESOLVERS = {
    "openbd": _openbd_cover_url,
    "hanmoto": _hanmoto_cover_url,
    "google": _google_books_cover_url,
}


def fetch_cover_image(
    isbn: str,
    *,
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    cache_dir: str | None = None,
    min_long_edge: int = DEFAULT_MIN_LONG_EDGE,
    opener=None,
    throttle_sec: float = 0.0,
) -> CoverResult | None:
    """Fetch a cover image for *isbn* across *sources* with optional disk cache."""
    isbn13 = isbn10_to_13(isbn)

    if cache_dir:
        cache_path = Path(cache_dir) / f"{isbn13}.jpg"
        if cache_path.is_file():
            data = cache_path.read_bytes()
            img = _decode_image(data)
            if img is not None and max(img.shape[:2]) >= min_long_edge:
                return CoverResult(isbn13, img, "cache", str(cache_path), cached=True)

    for idx, src_name in enumerate(sources):
        resolver = _SOURCE_RESOLVERS.get(src_name)
        if resolver is None:
            raise ValueError(f"unknown cover source: {src_name}")
        if idx > 0 and throttle_sec > 0:
            time.sleep(throttle_sec)
        try:
            img_url = resolver(isbn13, opener=opener)
            if not img_url:
                continue
            raw = _http_get(img_url, opener=opener)
            img = _decode_image(raw)
        except Exception as e:
            logger.debug("cover fetch failed (%s, %s): %s", src_name, isbn13, e)
            continue
        if img is None:
            continue
        if max(img.shape[:2]) < min_long_edge:
            logger.info("skipping tiny cover from %s (%dx%d)", src_name, img.shape[1], img.shape[0])
            continue
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            (Path(cache_dir) / f"{isbn13}.jpg").write_bytes(raw)
        return CoverResult(isbn13, img, src_name, img_url, cached=False)

    return None


# --- Upscaling (without margins) ------------------------------------------------

def _unsharp_mask(img_bgr: np.ndarray, sigma: float = 1.0, amount: float = 0.35) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    return cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)


def _onnx_upscale_tiled(
    img_bgr: np.ndarray,
    model_path: str,
    tile_size: int = 256,
    overlap: int = 16,
) -> np.ndarray:
    """Run an NCHW float32 [0,1] super-resolution ONNX model with tiled blending."""
    import onnxruntime as ort

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w, _ = rgb.shape

    # Probe scale factor on a small patch
    probe = np.transpose(rgb[:min(h, 32), :min(w, 32)], (2, 0, 1))[None, ...]
    probe_out = session.run([out_name], {in_name: probe})[0]
    scale = max(1, round(probe_out.shape[2] / probe.shape[2]))

    out_acc = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    weight_acc = np.zeros((h * scale, w * scale, 1), dtype=np.float32)

    step = max(1, tile_size - overlap)
    for y0 in range(0, h, step):
        y1 = min(h, y0 + tile_size)
        y0_eff = max(0, y1 - tile_size)
        for x0 in range(0, w, step):
            x1 = min(w, x0 + tile_size)
            x0_eff = max(0, x1 - tile_size)

            patch = np.transpose(rgb[y0_eff:y1, x0_eff:x1], (2, 0, 1))[None, ...]
            pred = session.run([out_name], {in_name: patch})[0][0]
            pred_hwc = np.clip(np.transpose(pred, (1, 2, 0)), 0.0, 1.0)

            oy0, oy1 = y0_eff * scale, y1 * scale
            ox0, ox1 = x0_eff * scale, x1 * scale
            out_acc[oy0:oy1, ox0:ox1] += pred_hwc
            weight_acc[oy0:oy1, ox0:ox1] += 1.0

    merged = np.clip((out_acc / np.maximum(weight_acc, 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)


def upscale_cover(
    img_bgr: np.ndarray,
    *,
    target_long_edge: int = 2400,
    mode: str = "auto",
    model_path: str | None = None,
) -> tuple[np.ndarray, str]:
    """Upscale *img_bgr* toward *target_long_edge* without adding margins.

    *mode*: ``auto`` | ``onnx`` | ``lanczos`` | ``none``.
    """
    if mode == "none":
        return img_bgr, "none"

    h, w = img_bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge >= int(target_long_edge * 0.8):
        return img_bgr, "original"

    if mode in ("auto", "onnx") and model_path and os.path.isfile(model_path):
        try:
            sr = _onnx_upscale_tiled(img_bgr, model_path)
            sr_long = max(sr.shape[:2])
            if sr_long > target_long_edge:
                ratio = target_long_edge / float(sr_long)
                sr = cv2.resize(
                    sr,
                    (max(1, round(sr.shape[1] * ratio)), max(1, round(sr.shape[0] * ratio))),
                    interpolation=cv2.INTER_AREA,
                )
            elif sr_long < int(target_long_edge * 0.85):
                ratio = target_long_edge / float(sr_long)
                sr = cv2.resize(
                    sr,
                    (max(1, round(sr.shape[1] * ratio)), max(1, round(sr.shape[0] * ratio))),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            return sr, "onnx"
        except Exception as e:
            if mode == "onnx":
                raise
            logger.warning("ONNX upscale failed, falling back to Lanczos: %s", e)

    ratio = target_long_edge / float(long_edge)
    new_w = max(1, round(w * ratio))
    new_h = max(1, round(h * ratio))
    enlarged = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return _unsharp_mask(enlarged), "lanczos+usm"


# --- PDF assembly ---------------------------------------------------------------

def _median_body_box_pt(pdf, start_idx: int = 0, sample_limit: int = 10) -> tuple[float, float]:
    """Compute representative (width_pt, height_pt) from body pages."""
    ws, hs = [], []
    for page in list(pdf.pages)[start_idx:start_idx + sample_limit]:
        mb = [float(v) for v in page.mediabox]
        w, h = abs(mb[2] - mb[0]), abs(mb[3] - mb[1])
        if w > 0 and h > 0:
            ws.append(w)
            hs.append(h)
    if not ws:
        raise ValueError("PDF has no valid pages to infer body page size")
    return float(statistics.median(ws)), float(statistics.median(hs))


def _make_cover_page_pdf(cover_bgr: np.ndarray, w_pt: float, h_pt: float, jpeg_quality: int) -> bytes:
    """Create a single-page PDF with *cover_bgr* filling (w_pt, h_pt) with zero margins."""
    import pikepdf

    rgb = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    jpg_buf = BytesIO()
    pil.save(jpg_buf, format="JPEG", quality=jpeg_quality, optimize=True)
    jpg_bytes = jpg_buf.getvalue()

    img_h, img_w = cover_bgr.shape[:2]
    doc = pikepdf.new()
    xobj = pikepdf.Stream(
        doc,
        jpg_bytes,
        Type=pikepdf.Name("/XObject"),
        Subtype=pikepdf.Name("/Image"),
        Width=img_w,
        Height=img_h,
        ColorSpace=pikepdf.Name("/DeviceRGB"),
        BitsPerComponent=8,
        Filter=pikepdf.Name("/DCTDecode"),
    )
    content = pikepdf.Stream(doc, f"q {w_pt:.3f} 0 0 {h_pt:.3f} 0 0 cm /Im0 Do Q".encode("ascii"))
    page_dict = pikepdf.Dictionary(
        Type=pikepdf.Name("/Page"),
        MediaBox=pikepdf.Array([0, 0, round(w_pt, 3), round(h_pt, 3)]),
        Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=xobj)),
        Contents=content,
    )
    doc.pages.append(pikepdf.Page(page_dict))
    out_buf = BytesIO()
    doc.save(out_buf)
    return out_buf.getvalue()


def _make_blank_page_pdf(w_pt: float, h_pt: float) -> bytes:
    import pikepdf

    doc = pikepdf.new()
    page_dict = pikepdf.Dictionary(
        Type=pikepdf.Name("/Page"),
        MediaBox=pikepdf.Array([0, 0, round(w_pt, 3), round(h_pt, 3)]),
    )
    doc.pages.append(pikepdf.Page(page_dict))
    out_buf = BytesIO()
    doc.save(out_buf)
    return out_buf.getvalue()


def insert_cover_pages(
    src_pdf: str,
    dst_pdf: str,
    cover_bgr: np.ndarray,
    *,
    isbn: str | None = None,
    source_name: str = "manual",
    blank_back: bool = True,
    replace_cover: bool = False,
    force: bool = False,
    upscale_mode: str = "auto",
    upscale_model: str | None = None,
    target_dpi: int = DEFAULT_TARGET_DPI,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> CoverInsertResult:
    """Prepend a cover page (and optional blank verso) to *src_pdf* -> *dst_pdf*.

    The cover page keeps its own aspect ratio matched to the body height so
    there are no added margins and no cropping/distortion.
    """
    import pikepdf
    from .model import ensure_parent_dir, refuse_overwrite

    refuse_overwrite(src_pdf, dst_pdf)
    ensure_parent_dir(dst_pdf)

    with pikepdf.open(src_pdf) as pdf:
        if len(pdf.pages) == 0:
            raise ValueError(f"{src_pdf}: PDF has 0 pages")

        already = pdf.docinfo.get(COVER_META_KEY)
        if already is not None and not force and not replace_cover:
            return CoverInsertResult(
                src_pdf=src_pdf,
                dst_pdf=dst_pdf,
                isbn=isbn,
                source=source_name,
                cover_size_px=(cover_bgr.shape[1], cover_bgr.shape[0]),
                upscale_method="skipped",
                cover_box_pt=(0.0, 0.0),
                blank_back=blank_back,
                skipped=True,
                note=f"already has cover ({already})",
            )

        body_start = 1 if (replace_cover and len(pdf.pages) > 1) else 0
        body_w_pt, body_h_pt = _median_body_box_pt(pdf, start_idx=body_start)

        target_long_edge = max(1200, round(body_h_pt * target_dpi / 72.0))
        up_bgr, up_method = upscale_cover(
            cover_bgr,
            target_long_edge=target_long_edge,
            mode=upscale_mode,
            model_path=upscale_model,
        )

        img_h, img_w = up_bgr.shape[:2]
        cover_h_pt = round(body_h_pt, 3)
        cover_w_pt = round(body_h_pt * (img_w / float(img_h)), 3)

        cover_pdf_bytes = _make_cover_page_pdf(up_bgr, cover_w_pt, cover_h_pt, jpeg_quality)
        with pikepdf.open(BytesIO(cover_pdf_bytes)) as cpdf:
            if replace_cover:
                del pdf.pages[0]
            pdf.pages.insert(0, cpdf.pages[0])

        if blank_back:
            blank_bytes = _make_blank_page_pdf(round(body_w_pt, 3), round(body_h_pt, 3))
            with pikepdf.open(BytesIO(blank_bytes)) as bpdf:
                pdf.pages.insert(1, bpdf.pages[0])

        pdf.docinfo[COVER_META_KEY] = isbn or source_name
        pdf.save(dst_pdf, deterministic_id=True)

    return CoverInsertResult(
        src_pdf=src_pdf,
        dst_pdf=dst_pdf,
        isbn=isbn,
        source=source_name,
        cover_size_px=(img_w, img_h),
        upscale_method=up_method,
        cover_box_pt=(cover_w_pt, cover_h_pt),
        blank_back=blank_back,
        skipped=False,
    )
