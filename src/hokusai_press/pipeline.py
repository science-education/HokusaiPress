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
    Box,
    Document,
    Flag,
    Margin,
    PageKind,
    PageParams,
    ReviewStatus,
)
from .ocr.factory import DEFAULT_OCR_ENGINE
from .store import Store

MIN_CONTENT_AREA_FRAC = 0.12   # content smaller than this share of the page =
#                                detection failed -> use the whole page + flag
BLANK_INK_FRAC = 0.001         # below this ink fraction (with no OCR content) =
#                                a blank / show-through page -> render white


def compute_margin(original: np.ndarray, sk, regions, dpi=None) -> tuple[np.ndarray, Margin]:
    """Region-based shadow removal + content box for one page.

    Shared by analyze_document (OCR path) and recompute_shadows_and_margins
    (OCR-free fast path) so a margin/shadow-detection fix never needs OCR
    re-run to take effect -- regions are already known in both cases.

    find_content_box supplies the confidence signal; the box itself is
    margin.content_ink_box (all print minus recognisable scan garbage, plus
    the regions).
    """
    original = margin_mod.remove_region_shadows(original, regions)
    mg = margin_mod.find_content_box(original, sk)
    # Content = every printed mark that is not recognisably scan garbage
    # (ScanTailor-style, inclusive), plus the OCR/layout regions. Whatever lies
    # outside is whitened at render time, so this box must not miss real print
    # that OCR does not report as text (a page number on a black tab, a rule,
    # an ornament, part of a running head).
    box = margin_mod.content_ink_box(original, sk, regions)
    if box is not None:
        mg.content = box
    if regions:
        # A region is real OCR/layout-detected content -- trust it however
        # small (a lone part-title line on a divider page is deliberate).
        mg.confidence = max(mg.confidence, 1.0)
    # detection-failed safety: if there's no region AND the page covers too
    # little of the page (near-blank page, or detection missed an undetected
    # figure that no region model classified), a sliver crop would drop real
    # content. Fall back to the whole (shadow-removed) page and flag it, so
    # nothing is cut and a human can check.
    ph, pw = original.shape[:2]
    c = mg.content
    if not regions and (not c or c.width * c.height < MIN_CONTENT_AREA_FRAC * pw * ph):
        mg.content = Box(0.0, 0.0, float(pw), float(ph))
        mg.confidence = 0.0           # -> MARGIN_NOT_FOUND via align_margins
    mg.page_w, mg.page_h = float(pw), float(ph)
    import cv2
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original
    mg.bg_tint_frac = round(float(np.mean((gray >= 100) & (gray <= 245))), 4)
    glyphs = margin_mod.glyph_components(original, sk, dpi if dpi and dpi > 0 else 600)
    if glyphs:
        mg.glyph_box = Box(min(b.x0 for b in glyphs), min(b.y0 for b in glyphs),
                           max(b.x1 for b in glyphs), max(b.y1 for b in glyphs))
    return original, mg


@dataclass
class AnalyzeResult:
    document: Document
    originals: list[np.ndarray]
    warnings: list[str] = None  # type: ignore[assignment]
    profile: dict = field(default_factory=dict)  # stage -> seconds

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def _prep_page(args):
    """CPU-only half of Phase B: deskew detection + upright (~0.24s, measured
    on tmp0613). Split out from _finish_page so the NPU pipeline (see
    _analyze_pages_npu_pipelined) can run this for the NEXT page on a
    background thread while the CURRENT page's OCR call -- the only part
    that touches the NPU -- is in flight, instead of the NPU sitting idle
    while the main thread deskews."""
    source, original, ocr_img, dpi = args[0], args[1], args[2], args[3]
    sk = deskew_mod.find_skew(ocr_img)
    ocr_up = _apply_deskew(ocr_img, sk.angle_deg)
    return sk, ocr_up


def _finish_page(args, sk, ocr_up) -> tuple[PageParams, np.ndarray, Margin]:
    """OCR + region-based shadow removal/content box/blank check, given this
    page's already-computed deskew (see _prep_page). The OCR engine call
    inside content_mod.analyze is the one step that must never overlap
    another in-flight call on an NPU-class device (concurrent NPU inference
    requests crash the Intel NPU driver -- see analyze_document)."""
    (source, original, ocr_img, dpi, reoriented, model_dir, device,
     ocr_engine, layout_engine, text_engine, use_ocr, layout_provider,
     openvino_cache_dir, runtime) = args

    regions, cflags = content_mod.analyze(
        original, ocr_up, source, model_dir=model_dir,
        device=device, ocr_engine=ocr_engine, layout_engine=layout_engine,
        text_engine=text_engine, use_ocr=use_ocr,
        layout_provider=layout_provider,
        openvino_cache_dir=openvino_cache_dir, runtime=runtime,
    )

    original, mg = compute_margin(original, sk, regions, dpi)

    ocr_succeeded = use_ocr and (Flag.OCR_FAILED not in cflags)
    blank = False
    if ocr_succeeded and not regions:
        import cv2
        from .geometry.margin import ink_threshold, remove_edge_shadows
        gg = cv2.cvtColor(remove_edge_shadows(ocr_img), cv2.COLOR_BGR2GRAY)
        blank = float((gg <= ink_threshold(gg)).mean()) < BLANK_INK_FRAC

    page_flags = [f for f in cflags if not (blank and f == Flag.NO_TEXT)]
    if source.page_index in reoriented:
        page_flags.append(Flag.PAGE_REORIENTED)
    from .streak import page_streaks
    streaks = page_streaks(ocr_img, source.rotate)   # dust on the scanner
    if streaks:
        page_flags.append(Flag.SCANNER_STREAK)
    params = PageParams(
        source=source, dpi=dpi, deskew=sk, margin=mg, regions=regions,
        page_kind=PageKind.AUTO, flags=page_flags, blank=blank,
    )
    params._streaks = streaks          # for the book's warning (not stored)
    return params, original, mg


def _analyze_one_page(args) -> tuple[PageParams, np.ndarray, Margin]:
    """Per-page Phase B body (steps 1-4 + blank check). Pure function of its
    arguments (no shared mutable state) so it can run on a thread pool --
    the OCR engine is documented thread-safe (one shared session, lazily
    built once under a lock; HybridOCR.__call__ only reads + ORT run()), and
    everything else here works on this page's own local arrays."""
    sk, ocr_up = _prep_page(args)
    return _finish_page(args, sk, ocr_up)


def _analyze_pages_npu_pipelined(worker_args):
    """Deskew-prefetch pipeline for an NPU-class device: exactly ONE OCR call
    is ever in flight (the NPU driver crashes under concurrent inference
    requests -- see analyze_document), but a background thread runs the NEXT
    page's deskew (CPU-only) while the CURRENT page's OCR call (NPU-bound,
    ~13x longer than deskew on tmp0613) is running, so the NPU is never
    sitting idle waiting on deskew once it's ready for the next page."""
    import queue
    import threading

    prep_q: "queue.Queue" = queue.Queue(maxsize=2)
    stop = threading.Event()

    def _put_until_stopped(msg) -> bool:
        while not stop.is_set():
            try:
                prep_q.put(msg, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def producer():
        try:
            for args in worker_args:
                if stop.is_set():
                    return
                item = (args, *_prep_page(args))
                if not _put_until_stopped(("ok", item)):
                    return
        except Exception as exc:
            _put_until_stopped(("error", exc))
        else:
            _put_until_stopped(("done", None))

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    try:
        results = []
        while True:
            kind, payload = prep_q.get()
            if kind == "done":
                break
            if kind == "error":
                raise payload
            args, sk, ocr_up = payload
            results.append(_finish_page(args, sk, ocr_up))
        return results
    finally:
        stop.set()
        th.join()


def analyze_document(
    path: str,
    model_dir: str = "models",
    device: str = "auto",
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    layout_engine: str | None = None,
    text_engine: str | None = None,
    use_ocr: bool = True,
    learned_model: Optional[dict] = None,
    layout_provider=None,
    openvino_cache_dir: Optional[str] = None,
    runtime: str | None = "paddle",
    pages: "set[int] | None" = None,
    max_workers: int = 4,
    keep_orientation: bool = False,
    stored_pages: "dict[int, PageParams] | list[PageParams] | None" = None,
    align_nombre: bool = True,
) -> AnalyzeResult:
    from .source import load_page

    import cv2

    doc = Document(source_path=path)
    originals: list[np.ndarray] = []
    margins = []
    prof: dict = defaultdict(float)

    # Validate / build the OCR engine up front before loading pages or writing
    # outputs so missing OCR dependencies fail fast with an actionable error.
    if use_ocr:
        from .ocr import get_ocr_engine

        try:
            get_ocr_engine(
                ocr_engine,
                model_dir=model_dir,
                device=device,
                openvino_cache_dir=openvino_cache_dir,
                runtime=runtime,
                layout_engine=layout_engine,
                text_engine=text_engine,
            )
        except ImportError as exc:
            raise content_mod.wrap_ocr_import_error(exc, ocr_engine) from exc

    # Phase A: load every page first. A document-level binarization valley needs
    # all pages in hand; originals are retained and returned anyway, so this adds
    # no peak memory beyond the work rasters.
    t = perf_counter()
    loaded = list(load_page(path, pages))
    prof["raster"] += perf_counter() - t

    # One page orientation per book (a uniform page size is meaningless across
    # mixed portrait/landscape pages). A page whose image disagrees with the
    # book is first cropped to the sheet if the scan kept the gray scanner bed
    # around it (an upright black chapter page saved wide); only a sheet that
    # still disagrees was stored sideways, and it is turned the way its text
    # reads (orientation.choose_rotation). keep_orientation leaves pages as
    # stored. Both are recorded in SourceRef (paper, rotate) so every later
    # load (rebuild, remargin, review UI) sees the same page.
    from . import orientation as orient_mod
    from .source import (
        find_paper, find_paper_by_book_size, orient, pdf_page_sizes,
    )

    page_sizes = pdf_page_sizes(path) if pages is not None else []
    total_pages: int | None = (
        len(loaded) if pages is None else (len(page_sizes) if page_sizes else None)
    )
    stored_map: dict[int, PageParams] | None = None
    if pages is not None and stored_pages:
        cand_map = (
            dict(stored_pages)
            if isinstance(stored_pages, dict)
            else {p.source.page_index: p for p in stored_pages}
        )
        if total_pages is not None and total_pages > 0:
            if all(i in cand_map for i in range(total_pages)):
                stored_map = cand_map
        elif (
            len(cand_map) > len(loaded)
            and set(cand_map.keys()) == set(range(len(cand_map)))
        ):
            stored_map = cand_map
            total_pages = len(cand_map)

    reoriented: set[int] = set()
    if pages is not None and stored_map is not None:
        fixed = []
        for s, o, w, d in loaded:
            sp = stored_map.get(s.page_index)
            if sp is not None:
                s.paper = list(sp.source.paper) if sp.source.paper is not None else None
                s.rotate = 0 if keep_orientation else int(sp.source.rotate or 0)
                if s.rotate:
                    reoriented.add(s.page_index)
                if s.paper or s.rotate:
                    o = orient(o, s)
                    w = orient(w, s, s.ocr_scale)
            fixed.append((s, o, w, d))
        loaded = fixed
    else:
        is_landscape = [o.shape[1] > o.shape[0] for (_s, o, _w, _d) in loaded]
        dominant_landscape: bool | None = None
        if pages is not None and len(page_sizes) >= 3:
            all_landscape = [pw > ph for pw, ph in page_sizes]
            dominant_landscape = sum(all_landscape) * 2 > len(all_landscape)
        elif len(loaded) >= 3:
            dominant_landscape = sum(is_landscape) * 2 > len(loaded)

        if dominant_landscape is not None:
            book_pt = _book_median_pt(page_sizes, loaded, dominant_landscape)
            fixed = []
            engine = None
            for (s, o, w, d), landscape in zip(loaded, is_landscape):
                raw_h, raw_w = o.shape[:2]
                exp_wh = _expected_px(book_pt, d)
                box = None
                used_size_fallback = False
                if landscape != dominant_landscape:
                    box = find_paper(o)
                    if box is not None:
                        pw, ph = box[2] - box[0], box[3] - box[1]
                        cw, ch = (ph, pw) if ((pw > ph) != dominant_landscape) else (pw, ph)
                    else:
                        cw, ch = (raw_h, raw_w)
                else:
                    cw, ch = float(raw_w), float(raw_h)

                if exp_wh is not None:
                    exp_w, exp_h = exp_wh
                    if box is None:
                        cur_w, cur_h = float(raw_w), float(raw_h)
                    else:
                        pw, ph = box[2] - box[0], box[3] - box[1]
                        cur_w, cur_h = (pw, ph) if ((pw > ph) == (exp_w > exp_h)) else (ph, pw)
                    if _needs_paper_size_fallback(raw_w, raw_h, cur_w, cur_h, exp_w, exp_h):
                        fb = find_paper_by_book_size(o, exp_w, exp_h)
                        if fb is not None:
                            box = fb
                            used_size_fallback = True

                if box is not None:
                    s.paper = box
                    pw, ph = box[2] - box[0], box[3] - box[1]
                    landscape = pw > ph

                if (landscape != dominant_landscape or used_size_fallback) and not keep_orientation:
                    cands = (0, 180) if landscape == dominant_landscape else (90, 270)
                    if use_ocr:
                        if engine is None:
                            from .ocr import get_ocr_engine

                            engine = get_ocr_engine(
                                ocr_engine, model_dir=model_dir, device=device,
                                openvino_cache_dir=openvino_cache_dir,
                                runtime=runtime, layout_engine=layout_engine,
                                text_engine=text_engine)
                        s.rotate, _ = orient_mod.choose_rotation(
                            orient(w, s, s.ocr_scale), engine, candidates=cands)
                    else:
                        s.rotate = cands[0]
                    if s.rotate:
                        reoriented.add(s.page_index)
                    o = orient(o, s)
                    w = orient(w, s, s.ocr_scale)
                elif box is not None:
                    o = orient(o, s)
                    w = orient(w, s, s.ocr_scale)
                fixed.append((s, o, w, d))
            loaded = fixed

    # 2-pass binarization valley: robust book-wide Otsu (median), so a
    # show-through page uses the book valley at render instead of a fixed floor.
    t = perf_counter()
    doc.render.ink_valley = margin_mod.document_ink_valley(
        cv2.cvtColor(o, cv2.COLOR_BGR2GRAY) if o.ndim == 3 else o
        for (_s, o, _w, _d) in loaded
    )
    prof["margin"] += perf_counter() - t

    # Phase B: per-page analysis (deskew, OCR/content, region-based shadow
    # removal, content box, blank check -- see _analyze_one_page). Binding/ADF
    # edge shadows are removed per page AFTER OCR, using the text region as
    # the boundary (margin.remove_region_shadows) -- a region-derived rule
    # with no fixed-position constant.
    #
    # Run on a thread pool: each page is otherwise-independent work, and the
    # OCR engine is the bottleneck (NPU/CPU inference dominates wall time --
    # see content.py's _get_ocr_engine) but leaves it under-saturated when
    # called one page at a time (measured ~1.7x throughput with 4 workers on
    # tmp0613, CPU device). max_workers=1 falls back to plain sequential
    # iteration (no thread pool at all) for an easy escape hatch / debugging.
    #
    # NPU-class devices never get a page-level thread pool, regardless of the
    # caller's request: concurrent OCR (inference) requests crashed the
    # Intel NPU driver outright (ZE_RESULT_ERROR_DEVICE_LOST, "device hung,
    # reset, was removed") under sustained multi-threaded load on tmp0613 --
    # every page after the crash then fails (each one safely flagged
    # OCR_FAILED, never silently fabricated, so no data corruption -- but the
    # whole rest of the run is wasted). A fresh process recovers fine (the
    # NPU itself wasn't permanently damaged), so this is a real driver
    # concurrency limitation, not a one-off fluke worth retrying around.
    # Instead they get the deskew-prefetch pipeline (_analyze_pages_npu_
    # pipelined): still exactly one OCR call in flight, but the NPU is kept
    # fed back-to-back instead of idling on the next page's CPU-only deskew
    # (measured ~0.24s deskew vs ~3.3s OCR on tmp0613 -- recovers most of
    # that gap "for free", with no risk to the NPU driver).
    npu_class = {"qnn", "npu", "openvino-auto", "mps"}
    is_npu = device.lower() in npu_class
    effective_workers = 1 if is_npu else max_workers
    t = perf_counter()
    worker_args = [
        (source, original, ocr_img, dpi, reoriented, model_dir, device,
         ocr_engine, layout_engine, text_engine, use_ocr, layout_provider,
         openvino_cache_dir, runtime)
        for source, original, ocr_img, dpi in loaded
    ]
    if is_npu and len(worker_args) > 1:
        results = _analyze_pages_npu_pipelined(worker_args)
    elif effective_workers > 1 and len(worker_args) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=effective_workers) as ex:
            results = list(ex.map(_analyze_one_page, worker_args))
    else:
        results = [_analyze_one_page(a) for a in worker_args]
    prof["phase_b"] += perf_counter() - t

    for params, original, mg in results:
        # deskew review is decided document-level below (book-relative angle
        # outliers), not per-page -- the per-page confidence isn't comparable
        # across books.
        doc.pages.append(params)
        originals.append(original)
        margins.append(mg)

    # deskew review: flag pages whose applied angle is a book-relative outlier
    for params, is_out in zip(
        doc.pages, deskew_mod.flag_skew_outliers([p.deskew for p in doc.pages])
    ):
        if is_out:
            params.flags.append(Flag.DESKEW_LOW_CONF)

    # 5. read page numbers from OCR and snap nombre to the real number (one
    # consistent band), so normalization anchors correctly and missing pages
    # can be detected. Runs before align/normalize so the better nombre feeds them.
    from . import nombre as nombre_mod
    from . import nombre_reader as nombre_reader_mod
    from .model import Region, RegionKind

    t = perf_counter()
    heights = [o.shape[0] for o in originals]
    widths = [o.shape[1] for o in originals]
    reader_enabled = os.environ.get("HOKUSAI_DISABLE_NOMBRE_READER") != "1"
    will_run_guided = (
        align_nombre
        and reader_enabled
        and pages is None
        and len(doc.pages) >= 2 * nombre_reader_mod.GUIDED_MIN_GROUP_PAGES
    )

    def _run_legacy_nombre_reader() -> None:
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

    if reader_enabled and not will_run_guided:
        _run_legacy_nombre_reader()
    warnings = nombre_mod.resolve(doc.pages, heights, widths, total_pages=total_pages)
    from .streak import streak_warning
    w = streak_warning({p.source.page_index: getattr(p, "_streaks", [])
                        for p in doc.pages})
    if w:
        warnings.append(w)

    # 6. cross-page margin consistency + 3-pass margin normalization.
    # A confirmed-blank page (rendered as pure white regardless of margin/
    # content -- see render_page_image's blank short-circuit) has nothing to
    # review: confidence=0.0 there just means "no ink to anchor on", not a
    # detection failure, so MARGIN_NOT_FOUND would only add noise to the queue.
    for params, flag in zip(doc.pages, margin_mod.align_margins(margins)):
        if flag is not None and not params.blank:
            params.flags.append(flag)
    tint_stats = [margin_mod.compute_page_tint_stats(img) for img in originals]
    for params, is_tinted in zip(doc.pages, margin_mod.detect_tinted_pages(doc.pages, tint_stats)):
        if params.margin is not None:
            params.margin.tinted = is_tinted
    if pages is not None and stored_map is not None:
        combined = dict(stored_map)
        for params in doc.pages:
            combined[params.source.page_index] = params
        norm_pages = [combined[idx] for idx in sorted(combined)]
    else:
        norm_pages = doc.pages

    if align_nombre:
        # Pass 1: type-area margin normalization
        margin_mod.normalize_margins_pass1(norm_pages, doc.render.output_margin_mm)
        # Pass 2a: Pass-1-guided margin-band folio localization
        if will_run_guided:
            t_folio = perf_counter()
            guided = nombre_reader_mod.localize_folios_guided(
                norm_pages,
                originals=originals,
                max_workers=effective_workers,
            )
            prof["folio_align"] += perf_counter() - t_folio
            if guided.get("applied"):
                nombre_mod.infer_folio_runs(norm_pages)
                gap_w = nombre_mod.refresh_gap_warnings(norm_pages)
                warnings = [msg for msg in warnings if "page(s) may be missing" not in msg] + gap_w
            else:
                # Fallback to per-page folio reader if guided pair didn't form
                _run_legacy_nombre_reader()
                gap_w = nombre_mod.resolve(doc.pages, heights, widths, total_pages=total_pages)
                warnings = [msg for msg in warnings if "page(s) may be missing" not in msg] + gap_w
        # Pass 2b + Pass 3: folio-aligned margin refinement (idempotent)
        margin_mod.normalize_margins(norm_pages, doc.render.output_margin_mm)
    else:
        margin_mod.normalize_margins_pass1(norm_pages, doc.render.output_margin_mm)
    _apply_edge_folio_protection(
        norm_pages,
        originals_by_idx={p.source.page_index: img for p, img in zip(doc.pages, originals)},
    )
    from .duplex import (
        apply_duplex_skew_exceptions,
        detect_blank_insertions,
        determine_duplex_pairing,
    )
    pairing = determine_duplex_pairing(norm_pages)
    if pairing.known:
        orig_by_idx = {p.source.page_index: img for p, img in zip(doc.pages, originals)}
        skew_changes = apply_duplex_skew_exceptions(
            norm_pages, pairing, lambda idx: orig_by_idx[idx],
        )
        if skew_changes:
            changed_idxs = {c["page_index"] for c in skew_changes}
            for p in doc.pages:
                if p.source.page_index not in changed_idxs:
                    continue
                img = orig_by_idx[p.source.page_index]
                if use_ocr:
                    ocr_up = _apply_deskew(img, p.deskew.angle_deg)
                    new_regs, _ = content_mod.analyze(
                        img, ocr_up, p.source, model_dir=model_dir,
                        device=device, ocr_engine=ocr_engine,
                        layout_engine=layout_engine, text_engine=text_engine,
                        use_ocr=use_ocr, layout_provider=layout_provider,
                        openvino_cache_dir=openvino_cache_dir, runtime=runtime,
                    )
                    p.regions = new_regs
                _, p.margin = compute_margin(img, p.deskew, p.regions, p.dpi)
            if align_nombre:
                margin_mod.normalize_margins(norm_pages, doc.render.output_margin_mm)
            else:
                margin_mod.normalize_margins_pass1(norm_pages, doc.render.output_margin_mm)
            _apply_edge_folio_protection(norm_pages, originals_by_idx=orig_by_idx)
        ins = detect_blank_insertions(norm_pages, pairing)
        doc.blank_insert_before = list(ins["insert_before"])
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


def check_doc_conflict(store: Store, path: str, doc_id: str | None = None) -> str | None:
    """Return an English error message if *doc_id* in *store* already refers to
    a different existing file on disk, else None."""
    if doc_id is None:
        doc_id = os.path.basename(path)
    from .model import same_path

    for row in store.list_pages(doc_id):
        stored_path = row.params.source.path
        if stored_path and not same_path(stored_path, path) and os.path.exists(stored_path):
            return (
                f"error: doc_id '{doc_id}' in DB already refers to existing file "
                f"'{stored_path}', which conflicts with '{path}'; "
                f"use a separate --db or rename one of the files"
            )
    return None


def run(
    path: str,
    out_pdf: str,
    db_path: str = "hokusai.db",
    model_dir: str = "models",
    device: str = "auto",
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    layout_engine: str | None = None,
    text_engine: str | None = None,
    use_ocr: bool = True,
    learned_model_path: Optional[str] = None,
    openvino_cache_dir: Optional[str] = None,
    runtime: str | None = "paddle",
    pages: "set[int] | None" = None,
    keep_orientation: bool = False,
    align_nombre: bool = True,
) -> dict:
    """Full batch run: analyze, persist params, render the PDF."""
    from . import learn
    from .model import ensure_parent_dir, refuse_overwrite
    from .render import build_pdf

    refuse_overwrite(path, out_pdf)
    ensure_parent_dir(out_pdf)
    doc_id = os.path.basename(path)
    existing_pages: dict[int, PageParams] | None = None
    if os.path.exists(db_path):
        store = Store(db_path)
        try:
            if err := check_doc_conflict(store, path, doc_id):
                raise ValueError(err)
            if pages is not None:
                existing_pages = {
                    r.page_index: r.params for r in store.list_pages(doc_id)
                }
        finally:
            store.close()

    learned = learn.load(learned_model_path) if learned_model_path else None
    result = analyze_document(path, model_dir=model_dir, device=device,
                              ocr_engine=ocr_engine,
                              layout_engine=layout_engine,
                              text_engine=text_engine,
                              use_ocr=use_ocr,
                              learned_model=learned,
                              openvino_cache_dir=openvino_cache_dir,
                              runtime=runtime, pages=pages,
                              keep_orientation=keep_orientation,
                              stored_pages=existing_pages,
                              align_nombre=align_nombre)
    t = perf_counter()
    store = Store(db_path)
    try:
        if pages is None:
            store.replace_doc_pages(doc_id, result.document.pages)
            # parameter-profile features (docs/PARAM_PROFILE_PLAN.md): geometry /
            # structure / statistics only, never OCR text. Only on a full run -- a
            # --pages subset must not overwrite the whole-book profile.
            _save_profile(store, doc_id, result)
            if result.document.blank_insert_before:
                meta = store.get_doc_meta(doc_id)
                meta["blank_insert_before"] = list(result.document.blank_insert_before)
                store.set_doc_meta(doc_id, meta)
        else:
            # key by the TRUE pdf page index (not enumeration) so a --pages subset
            # keeps real page numbers in the queue / UI / rebuild
            for params in result.document.pages:
                store.upsert_page(doc_id, params.source.page_index, params)
    finally:
        store.close()
    result.profile["db"] = perf_counter() - t

    t = perf_counter()
    build_pdf(result.document, result.originals, out_pdf)
    result.profile["render+pdf"] = perf_counter() - t
    flagged = [p.source.page_index for p in result.document.pages if p.needs_review()]
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
            db_path: str = "hokusai.db", max_workers: int = 1,
            use_processes: bool = False,
            pages: "set[int] | None" = None,
            margin_mode: str = "normalize") -> dict:
    """Regenerate the PDF purely from stored (possibly corrected) parameters.

    margin_mode "none" renders the whole scanned page (deskew + binarize only,
    no crop / margin fill / shadow removal); see RenderSettings.margin_mode.

    The output is a pure function of the parameters + the original image, so a
    review correction is realized simply by re-rendering — no image was ever
    destructively edited. This is the non-destructive model paying off.

    max_workers > 1 parallelizes both the per-page original-image load (see
    recompute_shadows_and_margins) and build_pdf's render+encode (the
    dominant cost -- see build_pdf).

    pages, if given, restricts the OUTPUT to that subset of 0-based page
    indices (e.g. a fast preview of a few pages) -- it does not affect
    what's stored; the full book's params are untouched.
    """
    from .model import refuse_overwrite
    from .render import build_pdf
    from .source import load_single

    refuse_overwrite(source_path, out_pdf)

    store = Store(db_path)
    try:
        rows = store.list_pages(doc_id)
        doc_meta = store.get_doc_meta(doc_id)
    finally:
        store.close()
    if pages is not None:
        rows = [r for r in rows if r.page_index in pages]
    if not rows:
        raise ValueError(f"no stored pages for doc_id={doc_id}")

    import cv2

    if margin_mode not in ("normalize", "none"):
        raise ValueError(f"margin_mode must be 'normalize' or 'none', not {margin_mode!r}")
    doc = Document(source_path=source_path)
    doc.render.margin_mode = margin_mode
    doc.pages = [row.params for row in rows]
    doc.blank_insert_before = list(doc_meta.get("blank_insert_before", []))
    load_args = [(source_path, row.params.source.page_index) for row in rows]
    if max_workers > 1 and len(load_args) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            originals = [o for o, _ in ex.map(
                lambda a: load_single(*a), load_args)]
    else:
        originals = [load_single(*a)[0] for a in load_args]
    # every stored box (deskew, margin, OCR regions) is in the frame analyze
    # saw: the sheet cropped from the scanner bed and turned upright (see
    # SourceRef.paper/rotate); without this such pages render wrong.
    from .source import orient

    originals = [orient(o, p.source) for o, p in zip(originals, doc.pages)]
    # rebuild makes a fresh Document, so the shadow-band model isn't carried in
    # the stored per-page params -- recompute it from the originals (same
    # document-level detector as analyze) so render applies the identical bands.
    doc.render.shadow_bands = margin_mod.detect_shadow_bands(
        cv2.cvtColor(o, cv2.COLOR_BGR2GRAY) if o.ndim == 3 else o for o in originals
    )
    doc.render.ink_valley = margin_mod.document_ink_valley(
        cv2.cvtColor(o, cv2.COLOR_BGR2GRAY) if o.ndim == 3 else o for o in originals
    )
    build_pdf(doc, originals, out_pdf, max_workers=max_workers,
              use_processes=use_processes)
    return {"doc_id": doc_id, "pages": len(rows), "out_pdf": out_pdf}


_RECOMPUTED_MARGIN_FLAGS = frozenset({
    Flag.MARGIN_NOT_FOUND,
    Flag.MARGIN_INCONSISTENT,
    Flag.PAGE_NUMBER_GAP,
    Flag.CONTENT_SCALED_DOWN,
})


def _sync_review_status(pages: list[PageParams]) -> None:
    """Update review_status for auto/needs_review pages from current flags.

    Human-decided pages (approved / corrected) are never overwritten.
    """
    for p in pages:
        if p.review_status in (ReviewStatus.AUTO, ReviewStatus.NEEDS_REVIEW):
            p.review_status = (
                ReviewStatus.NEEDS_REVIEW if p.flags else ReviewStatus.AUTO
            )


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
    _sync_review_status(pages)
    for p in pages:
        store.upsert_page(doc_id, p.source.page_index, p)
    return len(pages)


def _load_and_compute_margin(args):
    source_path, p = args
    from .source import load_single, orient

    original, _ = load_single(source_path, p.source.page_index)
    oriented = orient(original, p.source)
    tint_stats = margin_mod.compute_page_tint_stats(oriented)
    cleaned, mg = compute_margin(oriented, p.deskew, p.regions, p.dpi)
    h, w = cleaned.shape[:2]
    return h, w, mg, tint_stats


def _apply_edge_folio_protection(pages: list[PageParams],
                                 source_path: Optional[str] = None,
                                 originals_by_idx: Optional[dict[int, np.ndarray]] = None) -> None:
    """Fallback: for pages where normal folio search missed the folio, scan the
    outer band and translate crop (never scale) + record edge_folio_box."""
    from .source import load_single, orient

    for idx, p in enumerate(pages):
        if p.blank or not (p.margin and p.margin.crop) or getattr(p.margin, "tinted", False):
            continue
        if not margin_mod.folio_not_found(p):
            continue
        if originals_by_idx is not None:
            img = originals_by_idx.get(p.source.page_index)
            if img is None:
                continue
        elif source_path is not None:
            img, _ = load_single(source_path, p.source.page_index)
            img = orient(img, p.source)
        else:
            continue
        dpi = p.dpi if p.dpi and p.dpi > 0 else 600.0
        ef = margin_mod.detect_edge_folio_box(img, p, dpi)
        pf = margin_mod.detect_predicted_folio_box(img, pages, idx, dpi)
        if ef is not None and pf is not None:
            combined = margin_mod.Box(
                min(ef.x0, pf.x0), min(ef.y0, pf.y0),
                max(ef.x1, pf.x1), max(ef.y1, pf.y1),
            )
            margin_mod.protect_edge_folio(p, combined)
        elif ef is not None:
            margin_mod.protect_edge_folio(p, ef)
        elif pf is not None:
            margin_mod.protect_edge_folio(p, pf)


def _book_median_pt(page_sizes, loaded, dominant_landscape: bool) -> tuple[float, float] | None:
    """Median (w_pt, h_pt) of normal pages matching `dominant_landscape`."""
    import statistics

    pts = [
        (float(pw), float(ph)) for pw, ph in page_sizes
        if (pw > ph) == dominant_landscape and pw > 0 and ph > 0
    ]
    if len(pts) >= 3:
        return statistics.median(p[0] for p in pts), statistics.median(p[1] for p in pts)
    px_pts = [
        (o.shape[1] * 72.0 / d, o.shape[0] * 72.0 / d)
        for (_s, o, _w, d) in loaded
        if (o.shape[1] > o.shape[0]) == dominant_landscape and d and d > 0
    ]
    if len(px_pts) >= 3:
        return statistics.median(p[0] for p in px_pts), statistics.median(p[1] for p in px_pts)
    return None


def _expected_px(book_pt: tuple[float, float] | None, dpi: float | None) -> tuple[float, float] | None:
    if book_pt is None or not dpi or dpi <= 0:
        return None
    return book_pt[0] * dpi / 72.0, book_pt[1] * dpi / 72.0


def _needs_paper_size_fallback(
    raw_w: float, raw_h: float,
    cur_w: float, cur_h: float,
    exp_w: float, exp_h: float,
) -> bool:
    """True only when the raw scan is clearly larger than a normal book page
    AND the current paper/orientation dimensions do not match the book."""
    if exp_w <= 0 or exp_h <= 0:
        return False
    if raw_w * raw_h < 1.5 * exp_w * exp_h or max(raw_w / exp_w, raw_h / exp_h) < 1.4:
        return False
    if (cur_w > cur_h) != (exp_w > exp_h):
        return True
    return not (0.70 <= cur_w / exp_w <= 1.35 and 0.70 <= cur_h / exp_h <= 1.35)


def _fix_oversized_paper_pages(pages: list[PageParams], source_path: str) -> None:
    """Exception for remargin: re-detect paper & upright orientation (and remap
    stored OCR regions without re-running OCR) on oversized scans whose paper
    detection failed or yielded a wrong-aspect/wrong-size sheet."""
    import statistics
    from . import orientation as orient_mod
    from .model import SourceRef
    from .source import find_paper_by_book_size, load_single, orient, pdf_page_sizes

    normal = [
        p for p in pages
        if not p.blank and (p.source.paper is None)
        and p.margin and p.margin.page_w and p.margin.page_h
    ]
    if len(normal) < 3:
        return
    dpis = [p.dpi for p in normal if p.dpi and p.dpi > 0]
    med_dpi = statistics.median(dpis) if dpis else None
    if med_dpi is not None:
        pts = [
            (p.margin.page_w * 72.0 / (p.dpi or med_dpi),
             p.margin.page_h * 72.0 / (p.dpi or med_dpi))
            for p in normal
        ]
        book_w_pt = statistics.median(x[0] for x in pts)
        book_h_pt = statistics.median(x[1] for x in pts)
    else:
        book_w_px = statistics.median(p.margin.page_w for p in normal)
        book_h_px = statistics.median(p.margin.page_h for p in normal)
        book_w_pt = book_h_pt = None

    pdf_sizes = pdf_page_sizes(source_path)
    for p in pages:
        if not (p.margin and p.margin.page_w and p.margin.page_h):
            continue
        cur_w, cur_h = float(p.margin.page_w), float(p.margin.page_h)
        if med_dpi is not None and book_w_pt is not None:
            dpi_p = p.dpi if (p.dpi and p.dpi > 0) else med_dpi
            exp_w = book_w_pt * dpi_p / 72.0
            exp_h = book_h_pt * dpi_p / 72.0
        else:
            exp_w, exp_h = book_w_px, book_h_px

        # Fast pre-filter using PDF page size when available
        idx = p.source.page_index
        if pdf_sizes and idx < len(pdf_sizes) and book_w_pt is not None:
            rw_pt, rh_pt = pdf_sizes[idx]
            if rw_pt * rh_pt < 1.4 * book_w_pt * book_h_pt:
                continue

        raw_img, _ = load_single(source_path, idx)
        raw_h, raw_w = raw_img.shape[:2]
        if not _needs_paper_size_fallback(raw_w, raw_h, cur_w, cur_h, exp_w, exp_h):
            continue

        new_paper = find_paper_by_book_size(raw_img, exp_w, exp_h)
        if new_paper is None:
            continue
        pw = new_paper[2] - new_paper[0]
        ph = new_paper[3] - new_paper[1]
        same_orient = (pw > ph) == (exp_w > exp_h)
        cands = (0, 180) if same_orient else (90, 270)
        if p.regions:
            new_rot, _ = orient_mod.choose_rotation_from_regions(
                p.regions, (float(raw_w), float(raw_h)),
                p.source.paper, int(p.source.rotate or 0),
                float(p.deskew.angle_deg if p.deskew else 0.0),
                new_paper, candidates=cands,
            )
        else:
            new_rot = cands[0]

        new_ref = SourceRef(
            path=p.source.path, page_index=idx,
            embedded_image_id=p.source.embedded_image_id,
            ocr_scale=p.source.ocr_scale,
            paper=new_paper, rotate=new_rot,
        )
        new_oriented = orient(raw_img, new_ref)
        new_deskew = deskew_mod.find_skew(new_oriented)

        M = orient_mod.box_transform_matrix(
            (float(raw_w), float(raw_h)),
            p.source.paper, int(p.source.rotate or 0),
            float(p.deskew.angle_deg if p.deskew else 0.0),
            new_paper, new_rot, float(new_deskew.angle_deg),
        )
        for reg in p.regions:
            nx0, ny0, nx1, ny1 = orient_mod.map_box_coords(
                (reg.box.x0, reg.box.y0, reg.box.x1, reg.box.y1), M,
            )
            reg.box = Box(nx0, ny0, nx1, ny1)
        if p.margin.nombre_box is not None:
            nb = p.margin.nombre_box
            nx0, ny0, nx1, ny1 = orient_mod.map_box_coords(
                (nb.x0, nb.y0, nb.x1, nb.y1), M,
            )
            p.margin.nombre_box = Box(nx0, ny0, nx1, ny1)

        p.source.paper = new_paper
        p.source.rotate = new_rot
        p.deskew = new_deskew
        if new_rot and Flag.PAGE_REORIENTED not in p.flags:
            p.flags.append(Flag.PAGE_REORIENTED)
        elif not new_rot:
            p.flags = [f for f in p.flags if f != Flag.PAGE_REORIENTED]


def recompute_shadows_and_margins(store: Store, doc_id: str, source_path: str,
                                  output_margin_mm: float = 5.0,
                                  max_workers: int = 1,
                                  align_nombre: bool = True) -> int:
    """Re-run shadow removal + content-box detection from STORED regions, with
    no OCR / layout re-run. Returns the number of pages.

    For iterating on margin.py's shadow/content-box logic (the usual reason to
    touch this code): regions (OCR text/figure/photo boxes) don't depend on
    that logic, so re-deriving the margin from already-stored regions + a
    freshly loaded original is enough -- OCR is the expensive 90%+ of a full
    analyze_document() run, and this path skips it entirely. Follow with
    rebuild() to render the result; that already recomputes ink_valley/
    shadow_bands from the (now re-cleaned) originals.

    Preserves previously verified nombre_box / page_number / nombre_text so
    compute_margin's shape-only _detect_nombre heuristic never clobbers them.
    When align_nombre is True (default), runs Pass 1 -> Pass 2a guided folio
    re-identification -> Pass 2b/3 margin alignment.

    max_workers > 1 runs load_single + compute_margin for each page on a
    thread pool: pdfium extraction is already serialized process-wide (see
    source._PDFIUM_LOCK), and compute_margin's cv2 work releases the GIL, so
    pages still overlap profitably even though every load_single call shares
    that one lock.
    """
    rows = store.list_pages(doc_id)
    if err := check_doc_conflict(store, source_path, doc_id):
        raise ValueError(err)
    pages = [r.params for r in rows]
    _fix_oversized_paper_pages(pages, source_path)
    prev_nombre = [
        (
            p.margin.nombre_box if p.margin else None,
            p.margin.nombre_source if p.margin else None,
            p.page_number,
            p.nombre_text,
        )
        for p in pages
    ]

    worker_args = [(source_path, p) for p in pages]
    if max_workers > 1 and len(worker_args) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_load_and_compute_margin, worker_args))
    else:
        results = [_load_and_compute_margin(a) for a in worker_args]
    heights: list[int] = []
    widths: list[int] = []
    tint_stats = []
    has_preserved_nombre = False
    for p, (h, w, mg, ts), (prev_nb, prev_src, prev_pnum, prev_ntxt) in zip(
        pages, results, prev_nombre
    ):
        # Bug fix: never overwrite OCR/folio/manual (or legacy verified)
        # nombre_box / page_number / nombre_text with _detect_nombre's guess.
        if (
            prev_src in ("folio", "ocr", "manual")
            or prev_pnum is not None
            or prev_ntxt is not None
            or (prev_nb is not None and prev_src != "geom")
        ):
            mg.nombre_box = prev_nb
            mg.nombre_source = prev_src
            p.page_number = prev_pnum
            p.nombre_text = prev_ntxt
            has_preserved_nombre = True
        p.margin = mg
        heights.append(h)
        widths.append(w)
        tint_stats.append(ts)

    for p, is_tinted in zip(pages, margin_mod.detect_tinted_pages(pages, tint_stats)):
        if p.margin is not None:
            p.margin.tinted = is_tinted

    for p in pages:
        p.flags = [f for f in p.flags if f not in _RECOMPUTED_MARGIN_FLAGS]

    from . import nombre as nombre_mod
    from . import nombre_reader as nombre_reader_mod

    if not has_preserved_nombre:
        nombre_mod.resolve(pages, heights, widths)
    else:
        nombre_mod.refresh_gap_warnings(pages)

    for p, flag in zip(pages, margin_mod.align_margins([p.margin for p in pages])):
        if flag is not None and not p.blank and flag not in p.flags:
            p.flags.append(flag)

    if align_nombre:
        # Pass 1 -> Pass 2a (guided folio localization) -> Pass 2b/3
        margin_mod.normalize_margins_pass1(pages, output_margin_mm)
        if os.environ.get("HOKUSAI_DISABLE_NOMBRE_READER") != "1":
            guided = nombre_reader_mod.localize_folios_guided(
                pages, source_path=source_path, max_workers=max_workers,
            )
            if guided.get("applied"):
                nombre_mod.infer_folio_runs(pages)
                nombre_mod.refresh_gap_warnings(pages)
        margin_mod.normalize_margins(pages, output_margin_mm)
    else:
        margin_mod.normalize_margins_pass1(pages, output_margin_mm)
    _apply_edge_folio_protection(pages, source_path=source_path)

    from .duplex import (
        apply_duplex_skew_exceptions,
        detect_blank_insertions,
        determine_duplex_pairing,
    )
    pairing = determine_duplex_pairing(pages)
    if pairing.known:
        from .source import load_single, orient

        def _load_oriented(idx: int) -> np.ndarray:
            img, _ = load_single(source_path, idx)
            p_match = next(pp for pp in pages if pp.source.page_index == idx)
            return orient(img, p_match.source)

        skew_changes = apply_duplex_skew_exceptions(pages, pairing, _load_oriented)
        if skew_changes:
            changed_idxs = {c["page_index"] for c in skew_changes}
            for p in pages:
                if p.source.page_index in changed_idxs:
                    was_tinted = bool(p.margin and p.margin.tinted)
                    _, p.margin = _load_and_compute_margin((source_path, p))[1:3]
                    if p.margin is not None:
                        p.margin.tinted = was_tinted
            if align_nombre:
                margin_mod.normalize_margins(pages, output_margin_mm)
            else:
                margin_mod.normalize_margins_pass1(pages, output_margin_mm)
            _apply_edge_folio_protection(pages, source_path=source_path)
        ins = detect_blank_insertions(pages, pairing)
        meta = store.get_doc_meta(doc_id)
        meta["blank_insert_before"] = list(ins["insert_before"])
        store.set_doc_meta(doc_id, meta)

    _sync_review_status(pages)

    for p in pages:
        store.upsert_page(doc_id, p.source.page_index, p)
    return len(pages)
