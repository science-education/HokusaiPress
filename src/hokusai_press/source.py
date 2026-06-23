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
        ocr, scale = _downscale_for_ocr(img, None)
        yield SourceRef(path=path, ocr_scale=scale), img, ocr, None
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
    ext = os.path.splitext(path)[1].lower()
    if ext != ".pdf":
        return _imread_unicode(path), None
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        doc = pdfium.PdfDocument(path)
        try:
            original, dpi, _ = _extract_original(doc[page_index], doc, page_index)
        finally:
            doc.close()
    return original, dpi


_ROTATE_CV2 = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _extract_original(page, doc, index: int) -> tuple[np.ndarray, float | None, str | None]:
    """Prefer the single embedded full-page image (lossless original). Fall
    back to a 300 dpi rasterization when a page is not a simple scan."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    images = [obj for obj in page.get_objects()
              if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE]
    # get_size() already reflects /Rotate (it's the page's DISPLAY size), but
    # the embedded image bytes are in the page's own unrotated frame -- e.g. a
    # wide table scanned sideways and tagged /Rotate=270 so it displays as a
    # tall page (the reader turns the BOOK, not the table). Apply that same
    # rotation to the extracted bitmap (a lossless 90 deg permutation, not a
    # resample) so the original matches what every viewer/the rest of the
    # pipeline treats as "this page", instead of staying landscape while
    # every other page in the book is portrait.
    page_w_pt, page_h_pt = page.get_size()
    rotation = page.get_rotation() % 360

    if len(images) == 1:
        img_obj = images[0]
        try:
            pil = img_obj.get_bitmap(render=False).to_pil()
            arr = np.array(pil.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2_rot = _ROTATE_CV2.get(rotation)
            if cv2_rot is not None:
                bgr = cv2.rotate(bgr, cv2_rot)
            # dpi from embedded pixel size vs page point size (both now in
            # the same, rotation-applied orientation)
            dpi = bgr.shape[1] / (page_w_pt / 72.0) if page_w_pt else None
            return bgr, dpi, "img0"
        except Exception:
            pass

    bitmap = page.render(scale=OCR_DPI / 72)
    bgr = cv2.cvtColor(bitmap.to_numpy()[..., :3], cv2.COLOR_RGB2BGR)
    return bgr, float(OCR_DPI), None
