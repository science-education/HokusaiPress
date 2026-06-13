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
from typing import Optional

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
    learned_model: Optional[dict] = None,
    layout_provider=None,
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
            device=device, use_ocr=use_ocr, layout_provider=layout_provider,
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

    # 6. learned page-kind override (from accumulated review decisions)
    if learned_model:
        from . import learn

        for params in doc.pages:
            label, conf = learn.predict(learned_model, learn.page_features(params))
            if label and learn.is_confident(conf):
                params.page_kind = PageKind(label)
            elif Flag.KIND_BORDERLINE not in params.flags:
                params.flags.append(Flag.KIND_BORDERLINE)

    # 7. assign review status
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
    learned_model_path: Optional[str] = None,
) -> dict:
    """Full batch run: analyze, persist params, render the PDF."""
    from . import learn
    from .render import build_pdf

    learned = learn.load(learned_model_path) if learned_model_path else None
    result = analyze_document(path, model_dir=model_dir, device=device,
                              use_ocr=use_ocr, learned_model=learned)
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


def rebuild(doc_id: str, source_path: str, out_pdf: str,
            db_path: str = "hokusai.db") -> dict:
    """Regenerate the PDF purely from stored (possibly corrected) parameters.

    The output is a pure function of the parameters + the original image, so a
    review correction is realized simply by re-rendering — no image was ever
    destructively edited. This is the non-destructive model paying off.
    """
    from .render import build_pdf
    from .source import load_single

    store = Store(db_path)
    try:
        rows = store.list_pages(doc_id)
    finally:
        store.close()
    if not rows:
        raise ValueError(f"no stored pages for doc_id={doc_id}")

    doc = Document(source_path=source_path)
    originals = []
    for row in rows:
        doc.pages.append(row.params)
        original, _ = load_single(source_path, row.params.source.page_index)
        originals.append(original)
    build_pdf(doc, originals, out_pdf)
    return {"doc_id": doc_id, "pages": len(rows), "out_pdf": out_pdf}


def recompute_margins(store: Store, doc_id: str,
                      output_margin_mm: float = 5.0) -> int:
    """Re-run nombre-anchored margin normalization across a document's stored
    pages and persist the new crops. Returns the number of pages.

    Normalize-ONLY: detection (find_content_box / nombre) is never re-run, so
    manual content/nombre overrides are preserved. Because the output crop SIZE
    is the largest content extent in each parity group, editing one page's
    content/nombre changes the crop of its whole group — so the recompute spans
    every page. This is cheap (pure geometry over stored boxes; no image/OCR).
    """
    rows = store.list_pages(doc_id)
    pages = [r.params for r in rows]
    margin_mod.normalize_margins(pages, output_margin_mm)
    for p in pages:
        store.upsert_page(doc_id, p.source.page_index, p)
    return len(pages)
