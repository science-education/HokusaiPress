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


def compute_margin(original: np.ndarray, sk, regions) -> tuple[np.ndarray, Margin]:
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

    original, mg = compute_margin(original, sk, regions)

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
    from .source import find_paper, orient

    reoriented: set[int] = set()
    if len(loaded) >= 3:
        is_landscape = [o.shape[1] > o.shape[0] for (_s, o, _w, _d) in loaded]
        dominant_landscape = sum(is_landscape) * 2 > len(loaded)
        fixed = []
        engine = None
        for (s, o, w, d), landscape in zip(loaded, is_landscape):
            if landscape != dominant_landscape:
                box = find_paper(o)
                if box is not None:
                    s.paper = box
                    pw, ph = box[2] - box[0], box[3] - box[1]
                    landscape = pw > ph
                if landscape != dominant_landscape and not keep_orientation:
                    if use_ocr:
                        if engine is None:
                            from .ocr import get_ocr_engine

                            engine = get_ocr_engine(
                                ocr_engine, model_dir=model_dir, device=device,
                                openvino_cache_dir=openvino_cache_dir,
                                runtime=runtime, layout_engine=layout_engine,
                                text_engine=text_engine)
                        s.rotate, _ = orient_mod.choose_rotation(
                            orient(w, s, s.ocr_scale), engine)
                    else:
                        s.rotate = 90       # no OCR to tell: flagged for review
                    reoriented.add(s.page_index)
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
    from .streak import streak_warning
    w = streak_warning({p.source.page_index: getattr(p, "_streaks", [])
                        for p in doc.pages})
    if w:
        warnings.append(w)

    # 6. cross-page margin consistency + nombre-anchored normalization.
    # A confirmed-blank page (rendered as pure white regardless of margin/
    # content -- see render_page_image's blank short-circuit) has nothing to
    # review: confidence=0.0 there just means "no ink to anchor on", not a
    # detection failure, so MARGIN_NOT_FOUND would only add noise to the queue.
    for params, flag in zip(doc.pages, margin_mod.align_margins(margins)):
        if flag is not None and not params.blank:
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
) -> dict:
    """Full batch run: analyze, persist params, render the PDF."""
    from . import learn
    from .model import ensure_parent_dir, refuse_overwrite
    from .render import build_pdf

    refuse_overwrite(path, out_pdf)
    ensure_parent_dir(out_pdf)
    doc_id = os.path.basename(path)
    if os.path.exists(db_path):
        store = Store(db_path)
        try:
            if err := check_doc_conflict(store, path, doc_id):
                raise ValueError(err)
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
                              keep_orientation=keep_orientation)
    t = perf_counter()
    store = Store(db_path)
    try:
        if pages is None:
            store.replace_doc_pages(doc_id, result.document.pages)
            # parameter-profile features (docs/PARAM_PROFILE_PLAN.md): geometry /
            # structure / statistics only, never OCR text. Only on a full run -- a
            # --pages subset must not overwrite the whole-book profile.
            _save_profile(store, doc_id, result)
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


def _load_and_compute_margin(args):
    source_path, p = args
    from .source import load_single

    from .source import orient

    original, _ = load_single(source_path, p.source.page_index)
    return compute_margin(orient(original, p.source), p.deskew, p.regions)


def recompute_shadows_and_margins(store: Store, doc_id: str, source_path: str,
                                  output_margin_mm: float = 5.0,
                                  max_workers: int = 1) -> int:
    """Re-run shadow removal + content-box detection from STORED regions, with
    no OCR / layout re-run. Returns the number of pages.

    For iterating on margin.py's shadow/content-box logic (the usual reason to
    touch this code): regions (OCR text/figure/photo boxes) don't depend on
    that logic, so re-deriving the margin from already-stored regions + a
    freshly loaded original is enough -- OCR is the expensive 90%+ of a full
    analyze_document() run, and this path skips it entirely. Follow with
    rebuild() to render the result; that already recomputes ink_valley/
    shadow_bands from the (now re-cleaned) originals.

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
    worker_args = [(source_path, p) for p in pages]
    if max_workers > 1 and len(worker_args) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_load_and_compute_margin, worker_args))
    else:
        results = [_load_and_compute_margin(a) for a in worker_args]
    originals = []
    for p, (original, mg) in zip(pages, results):
        p.margin = mg
        originals.append(original)

    from . import nombre as nombre_mod

    heights = [o.shape[0] for o in originals]
    widths = [o.shape[1] for o in originals]
    nombre_mod.resolve(pages, heights, widths)
    for p, flag in zip(pages, margin_mod.align_margins([p.margin for p in pages])):
        if flag is not None and not p.blank and flag not in p.flags:
            p.flags.append(flag)
    margin_mod.normalize_margins(pages, output_margin_mm)

    for p in pages:
        store.upsert_page(doc_id, p.source.page_index, p)
    return len(pages)
