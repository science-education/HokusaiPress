"""Batch orchestration: ingest -> analyze (params only) -> flag -> store.

Per principle ①, this stage produces NO output images. It fills a
PageParams per page (deskew, margin, regions, page_kind) and decides which
pages are confident enough to render automatically vs. which go to the
review queue. Rendering happens later (render.build_pdf), reading the same
params, so a corrected page is just re-rendered.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import content as content_mod
from .geometry import deskew as deskew_mod
from .geometry import margin as margin_mod
from .model import (
    Document,
    Flag,
    PageKind,
    PageParams,
    ReviewStatus,
)
from .store import Store


@dataclass
class AnalyzeResult:
    document: Document
    originals: list[np.ndarray]


def analyze_document(
    path: str,
    model_dir: str = "models",
    device: str = "auto",
    use_ocr: bool = True,
) -> AnalyzeResult:
    from .source import load_page

    doc = Document(source_path=path)
    originals: list[np.ndarray] = []
    margins = []

    for source, original, ocr_img, dpi in load_page(path):
        # 1. deskew on the work raster (angle is scale-free)
        sk = deskew_mod.find_skew(ocr_img)
        # 2. deskew the work raster so OCR/margin see upright text
        ocr_up = _apply_deskew(ocr_img, sk.angle_deg)
        # 3. content separation (text/figure/photo) + OCR text
        regions, cflags = content_mod.analyze(
            original, ocr_up, source, model_dir=model_dir,
            device=device, use_ocr=use_ocr,
        )
        # 4. margin / nombre on the deskewed original
        mg = margin_mod.find_content_box(original, sk)

        params = PageParams(
            source=source, dpi=dpi, deskew=sk, margin=mg, regions=regions,
            page_kind=PageKind.AUTO, flags=list(cflags),
        )
        if not deskew_mod.is_confident(sk):
            params.flags.append(Flag.DESKEW_LOW_CONF)
        doc.pages.append(params)
        originals.append(original)
        margins.append(mg)

    # 5. cross-page margin consistency + nombre-anchored normalization
    for params, flag in zip(doc.pages, margin_mod.align_margins(margins)):
        if flag is not None:
            params.flags.append(flag)
    margin_mod.normalize_margins(doc.pages, doc.render.output_margin_mm)

    # 6. assign review status
    for params in doc.pages:
        params.review_status = (
            ReviewStatus.NEEDS_REVIEW if params.flags else ReviewStatus.AUTO
        )
    return AnalyzeResult(document=doc, originals=originals)


def _apply_deskew(img: np.ndarray, angle: float) -> np.ndarray:
    import cv2

    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderValue=(255, 255, 255))


def run(
    path: str,
    out_pdf: str,
    db_path: str = "hokusai.db",
    model_dir: str = "models",
    device: str = "auto",
    use_ocr: bool = True,
) -> dict:
    """Full batch run: analyze, persist params, render the PDF."""
    from .render import build_pdf

    result = analyze_document(path, model_dir=model_dir, device=device, use_ocr=use_ocr)
    doc_id = os.path.basename(path)
    store = Store(db_path)
    try:
        for i, params in enumerate(result.document.pages):
            store.upsert_page(doc_id, i, params)
    finally:
        store.close()

    build_pdf(result.document, result.originals, out_pdf)
    flagged = [i for i, p in enumerate(result.document.pages) if p.needs_review()]
    return {
        "doc_id": doc_id,
        "pages": len(result.document.pages),
        "needs_review": flagged,
        "out_pdf": out_pdf,
    }
