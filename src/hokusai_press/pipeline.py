"""Batch orchestration: ingest -> analyze (params only) -> flag -> store.

Per principle ①, this stage produces NO output images. It fills a
PageParams per page (deskew, margin, regions, page_kind) and decides which
pages are confident enough to render automatically vs. which go to the
review queue. Rendering happens later (render.build_pdf), reading the same
params, so a corrected page is just re-rendered.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter
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

MIN_CONTENT_AREA_FRAC = 0.12   # content smaller than this share of the page =
#                                detection failed -> use the whole page + flag
BLANK_INK_FRAC = 0.001         # below this ink fraction (with no OCR content) =
#                                a blank / show-through page -> render white


@dataclass
class AnalyzeResult:
    document: Document
    originals: list[np.ndarray]
    warnings: list[str] = None  # type: ignore[assignment]
    profile: dict = field(default_factory=dict)  # stage -> seconds

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def analyze_document(
    path: str,
    model_dir: str = "models",
    device: str = "auto",
    ocr_engine: str = "hybrid",
    layout_engine: str | None = None,
    text_engine: str | None = None,
    use_ocr: bool = True,
    learned_model: Optional[dict] = None,
    layout_provider=None,
    openvino_cache_dir: Optional[str] = None,
    runtime: str | None = "paddle",
    pages: "set[int] | None" = None,
) -> AnalyzeResult:
    from .source import load_page

    doc = Document(source_path=path)
    originals: list[np.ndarray] = []
    margins = []
    prof: dict = defaultdict(float)

    page_iter = load_page(path, pages)
    while True:
        t = perf_counter()
        try:
            source, original, ocr_img, dpi = next(page_iter)
        except StopIteration:
            break
        prof["raster"] += perf_counter() - t

        # 1-2. deskew on the work raster (angle is scale-free), then upright it
        t = perf_counter()
        sk = deskew_mod.find_skew(ocr_img)
        ocr_up = _apply_deskew(ocr_img, sk.angle_deg)
        prof["deskew"] += perf_counter() - t

        # 3. content separation (text/figure/photo) + OCR text
        t = perf_counter()
        regions, cflags = content_mod.analyze(
            original, ocr_up, source, model_dir=model_dir,
            device=device, ocr_engine=ocr_engine, layout_engine=layout_engine,
            text_engine=text_engine, use_ocr=use_ocr,
            layout_provider=layout_provider, openvino_cache_dir=openvino_cache_dir,
            runtime=runtime,
        )
        prof["ocr"] += perf_counter() - t

        # 4. margin / nombre on the deskewed original; expand the content box to
        # include detected regions so a figure/photo is never cropped out (the
        # ink box alone can be smaller than a photo). Shadows are not regions and
        # were already dropped from the ink box, so they stay excluded.
        t = perf_counter()
        mg = margin_mod.find_content_box(original, sk)
        from .model import Box
        if regions and mg.content:
            c = mg.content
            xs0 = [c.x0] + [r.box.x0 for r in regions]
            ys0 = [c.y0] + [r.box.y0 for r in regions]
            xs1 = [c.x1] + [r.box.x1 for r in regions]
            ys1 = [c.y1] + [r.box.y1 for r in regions]
            mg.content = Box(min(xs0), min(ys0), max(xs1), max(ys1))
        # detection-failed safety: if the content covers too little of the page
        # (near-blank page, or detection missed an undetected figure), a sliver
        # crop would drop real content. Fall back to the whole (shadow-removed)
        # page and flag it, so nothing is cut and a human can check.
        ph, pw = original.shape[:2]
        c = mg.content
        if not c or c.width * c.height < MIN_CONTENT_AREA_FRAC * pw * ph:
            mg.content = Box(0.0, 0.0, float(pw), float(ph))
            mg.confidence = 0.0           # -> MARGIN_NOT_FOUND via align_margins

        # OCR-confident blank: OCR ran, found NO content region (text/figure/
        # photo), and the (shadow-removed, clamp-thresholded) page has no real
        # ink -> it is genuinely empty (a blank leaf, or pure show-through), so
        # render it white. Gated by use_ocr so it can never erase real text, and
        # cross-checked against ink so an OCR-missed text page is NOT blanked.
        blank = False
        if use_ocr and not regions:
            import cv2
            from .geometry.margin import ink_threshold, remove_edge_shadows
            gg = cv2.cvtColor(remove_edge_shadows(ocr_img), cv2.COLOR_BGR2GRAY)
            blank = float((gg <= ink_threshold(gg)).mean()) < BLANK_INK_FRAC
        prof["margin"] += perf_counter() - t

        params = PageParams(
            source=source, dpi=dpi, deskew=sk, margin=mg, regions=regions,
            page_kind=PageKind.AUTO, flags=list(cflags), blank=blank,
        )
        if not deskew_mod.is_confident(sk):
            params.flags.append(Flag.DESKEW_LOW_CONF)
        doc.pages.append(params)
        originals.append(original)
        margins.append(mg)

    # 5. read page numbers from OCR and snap nombre to the real number (one
    # consistent band), so normalization anchors correctly and missing pages
    # can be detected. Runs before align/normalize so the better nombre feeds them.
    from . import nombre as nombre_mod
    from . import nombre_reader as nombre_reader_mod
    from .model import Region, RegionKind

    t = perf_counter()
    heights = [o.shape[0] for o in originals]
    widths = [o.shape[1] for o in originals]
    if os.environ.get("HOKUSAI_DISABLE_NOMBRE_READER") != "1":
        for params, original, h, w in zip(doc.pages, originals, heights, widths):
            if not params.margin or not params.margin.content:
                continue
            deskewed = _apply_deskew(original, params.deskew.angle_deg)
            extra = [
                r.box for r in params.regions
                if getattr(r, "layout_label", None) == "page_number"
            ]
            try:
                rcands = nombre_reader_mod.read_folio_candidates(
                    deskewed, params.margin.content, h, w, extra_boxes=extra)
            except Exception:
                rcands = []
            params._nombre_candidates = [
                (
                    c.band,
                    c.kind,
                    c.value,
                    Region(kind=RegionKind.TEXT, box=c.box, source="nombre_reader",
                           ocr_text=c.text, ocr_conf=c.conf),
                )
                for c in rcands
            ]
    warnings = nombre_mod.resolve(doc.pages, heights, widths)

    # 6. cross-page margin consistency + nombre-anchored normalization
    for params, flag in zip(doc.pages, margin_mod.align_margins(margins)):
        if flag is not None:
            params.flags.append(flag)
    margin_mod.normalize_margins(doc.pages, doc.render.output_margin_mm)
    prof["nombre+normalize"] += perf_counter() - t

    # 6. learned page-kind override (from accumulated review decisions)
    if learned_model:
        from . import learn

        for params in doc.pages:
            label, conf = learn.predict(learned_model, learn.page_features(params))
            if label and learn.is_confident(conf):
                params.page_kind = PageKind(label)
            elif Flag.KIND_BORDERLINE not in params.flags:
                params.flags.append(Flag.KIND_BORDERLINE)

    # 8. assign review status
    for params in doc.pages:
        params.review_status = (
            ReviewStatus.NEEDS_REVIEW if params.flags else ReviewStatus.AUTO
        )
    return AnalyzeResult(document=doc, originals=originals, warnings=warnings,
                         profile=dict(prof))


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
    ocr_engine: str = "hybrid",
    layout_engine: str | None = None,
    text_engine: str | None = None,
    use_ocr: bool = True,
    learned_model_path: Optional[str] = None,
    openvino_cache_dir: Optional[str] = None,
    runtime: str | None = "paddle",
    pages: "set[int] | None" = None,
) -> dict:
    """Full batch run: analyze, persist params, render the PDF."""
    from . import learn
    from .render import build_pdf

    learned = learn.load(learned_model_path) if learned_model_path else None
    result = analyze_document(path, model_dir=model_dir, device=device,
                              ocr_engine=ocr_engine,
                              layout_engine=layout_engine,
                              text_engine=text_engine,
                              use_ocr=use_ocr,
                              learned_model=learned,
                              openvino_cache_dir=openvino_cache_dir,
                              runtime=runtime, pages=pages)
    doc_id = os.path.basename(path)
    t = perf_counter()
    store = Store(db_path)
    try:
        # key by the TRUE pdf page index (not enumeration) so a --pages subset
        # keeps real page numbers in the queue / UI / rebuild
        for params in result.document.pages:
            store.upsert_page(doc_id, params.source.page_index, params)
        # parameter-profile features (docs/PARAM_PROFILE_PLAN.md): geometry /
        # structure / statistics only, never OCR text. Only on a full run -- a
        # --pages subset must not overwrite the whole-book profile.
        if pages is None:
            _save_profile(store, doc_id, result)
    finally:
        store.close()
    result.profile["db"] = perf_counter() - t

    t = perf_counter()
    build_pdf(result.document, result.originals, out_pdf)
    result.profile["render+pdf"] = perf_counter() - t
    flagged = [i for i, p in enumerate(result.document.pages) if p.needs_review()]
    return {
        "doc_id": doc_id,
        "pages": len(result.document.pages),
        "needs_review": flagged,
        "out_pdf": out_pdf,
        "warnings": result.warnings,
        "profile": result.profile,
    }


def _save_profile(store, doc_id: str, result) -> None:
    """Persist copyright-safe per-book features + a posterior snapshot. page
    features for every page (cheap geometry tier); region features only for a
    representative subset (flagged + sampled) per the (b) granularity."""
    import dataclasses
    import json as _json

    from . import profile as profile_mod

    pages = result.document.pages
    widths = [o.shape[1] for o in result.originals]
    heights = [o.shape[0] for o in result.originals]
    pfeats, rfeats = profile_mod.extract_features(pages, widths, heights)
    bp = profile_mod.aggregate(pfeats, rfeats, widths, heights)
    front_end, body_end = profile_mod.segment_structure(
        pfeats, bp.dominant_offset)

    rep = {i for i, p in enumerate(pages) if p.needs_review()}
    rep |= set(range(0, len(pages), 25))
    rep_pidx = {pages[i].source.page_index for i in rep}
    rep_rfeats = [rf for rf in rfeats if rf.page_index in rep_pidx]

    store.save_book(doc_id, title=doc_id, writing_dir=bp.writing_dir,
                    binding=bp.binding, source="scan")
    store.save_page_features(doc_id, pfeats)
    store.save_region_features(doc_id, rep_rfeats)
    store.save_scan_profile(doc_id, None, front_end, body_end, bp.page_count,
                            bp.ocr_pages, _json.dumps(dataclasses.asdict(bp)))


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
