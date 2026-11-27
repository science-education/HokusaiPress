"""HokusaiPress CLI."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import time

from .ocr.factory import DEFAULT_OCR_ENGINE


def _is_folder_out(out: str) -> bool:
    """--out denotes a folder when it is an existing directory, ends with a path
    separator, or has no extension."""
    return (os.path.isdir(out) or out.endswith(("/", "\\"))
            or os.path.splitext(out)[1] == "")


def _resolve_out(out: str, source: str, suffix: str = "", mkdir: bool = True) -> str:
    """Allow --out to be a folder: save as the source's name (+ suffix) with a
    .pdf extension inside it (folder created on demand when mkdir=True).
    Otherwise --out is used as the output file path verbatim."""
    if _is_folder_out(out):
        if mkdir:
            os.makedirs(out, exist_ok=True)
        stem = os.path.splitext(os.path.basename(source))[0]
        return os.path.join(out, stem + suffix + ".pdf")
    if mkdir:
        from .model import ensure_parent_dir

        ensure_parent_dir(out)
    return out


def _norm_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _check_doc_id_collisions(sources: list[str], db_path: str) -> str | None:
    """Return an English error message if multiple inputs in *sources* share the
    same basename (doc_id), or if any input's doc_id already exists in *db_path*
    pointing to a different existing file on disk."""
    from .pipeline import check_doc_conflict
    from .store import Store

    seen_basenames: dict[str, str] = {}
    for src in sources:
        doc_id = os.path.basename(src)
        key = os.path.normcase(doc_id)
        if key in seen_basenames:
            prev = seen_basenames[key]
            return (
                f"error: duplicate input filename '{doc_id}': '{prev}' and "
                f"'{src}' share the same doc_id and would collide in the DB; "
                f"run them with separate --db paths or rename one of the files"
            )
        seen_basenames[key] = src

    if os.path.exists(db_path):
        store = Store(db_path)
        try:
            for src in sources:
                if err := check_doc_conflict(store, src, os.path.basename(src)):
                    return err
        finally:
            store.close()
    return None


def _check_output_collisions(
    sources: list[str],
    planned_outputs: list[tuple[str, str]],
) -> str | None:
    """Return an English error message if any output collides with an input or
    another output, else None."""
    input_map: dict[str, str] = {}
    for src in sources:
        input_map.setdefault(_norm_path(src), src)

    seen_outputs: dict[str, tuple[str, str]] = {}
    for src, out_path in planned_outputs:
        key = _norm_path(out_path)
        if key in input_map:
            matched_src = input_map[key]
            return (
                f"error: output path '{out_path}' would overwrite input "
                f"'{matched_src}'; use --suffix or a different output folder"
            )
        if key in seen_outputs:
            prev_src, prev_out = seen_outputs[key]
            if prev_src != src:
                return (
                    f"error: output path collision: '{out_path}' is produced by "
                    f"both '{prev_src}' and '{src}'; use --suffix or a "
                    f"different output folder"
                )
            return (
                f"error: output path collision: '{out_path}' and '{prev_out}' "
                f"resolve to the same file for input '{src}'; use --suffix or "
                f"a different output folder"
            )
        seen_outputs[key] = (src, out_path)
    return None


def _expand_sources(paths: list[str]) -> list[str]:
    """Resolve each input path to concrete files: a directory expands to its
    *.pdf, a glob expands to its matches, a plain path is taken as-is. Results
    are de-duplicated while preserving order (PowerShell does not glob args for
    external programs, so the CLI does it)."""
    expanded: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            expanded.extend(sorted(glob.glob(os.path.join(p, "*.pdf"))))
        elif any(c in p for c in "*?["):
            expanded.extend(sorted(glob.glob(p)))
        else:
            expanded.append(p)
    seen, uniq = set(), []
    for p in expanded:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(p)
    return uniq


def _parse_pages(spec: "str | None") -> "set[int] | None":
    """Parse a page selection in PDF page numbers (1 = the first page), like
    '10', '53,237-238', '1-5', to the set of 0-based page indices the pipeline
    uses. Returns None for the whole document; ValueError for a bad spec."""
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
            else:
                a = b = int(part)
        except ValueError:
            raise ValueError(f"--pages: not a page or range: {part!r}") from None
        if a < 1 or b < a:
            raise ValueError(f"--pages: pages start at 1 (the PDF's first "
                             f"page) and ranges run low-high: {part!r}")
        out.update(range(a - 1, b))
    return out or None


def _run_one(payload: dict) -> dict:
    """Process a single source to its own db (picklable; used by workers and the
    serial path). Returns the run summary plus timing and the db it wrote."""
    from .pipeline import run

    t0 = time.perf_counter()
    summary = run(
        payload["src"], payload["out"], db_path=payload["db"],
        model_dir=payload["model_dir"], device=payload["device"],
        ocr_engine=payload["ocr_engine"],
        layout_engine=payload["layout_engine"],
        text_engine=payload["text_engine"],
        use_ocr=payload["use_ocr"],
        learned_model_path=payload["learned_model"],
        openvino_cache_dir=payload["cache"],
        runtime=payload["runtime"], pages=payload["pages"],
        keep_orientation=payload.get("keep_orientation", False),
        align_nombre=payload.get("align_nombre", True),
    )
    summary["tpb"] = time.perf_counter() - t0
    if payload.get("image_only"):
        # the same pages without the OCR text layer (the "binarized" stage)
        from .textlayer import image_only_copy

        image_only_copy(payload["out"], payload["image_only"])
        summary["image_only"] = payload["image_only"]
    return summary


def _print_summary(summary: dict, profile: bool) -> bool:
    """Print the [OK]/warn/prof lines for one file; return True if it has review."""
    pages = summary["pages"] or 1
    tpb = summary["tpb"]
    line = f"[OK] {summary['doc_id']}: {summary['pages']} pages -> {summary['out_pdf']}"
    flagged = bool(summary["needs_review"])
    if flagged:
        line += f"  (needs review: {len(summary['needs_review'])} pages)"
    line += f"  TPB={tpb:.1f}s TPP={tpb / pages:.2f}s/page"
    print(line, flush=True)
    if summary.get("image_only"):
        print(f"  [image-only] {summary['image_only']}", flush=True)
    for w in summary.get("warnings", []):
        print(f"  [warn] {w}")
    if profile:
        prof = summary.get("profile", {})
        total = sum(prof.values()) or 1.0
        for stage, sec in sorted(prof.items(), key=lambda kv: -kv[1]):
            print(f"  [prof] {stage:<16} {sec:7.1f}s "
                  f"{sec / pages:6.3f}s/page  {sec / total * 100:4.0f}%")
    return flagged


def _merge_worker_db(target_db: str, wdb: str, full_run: bool = True) -> None:
    """Fold one worker temp db's tables into *target_db*, then remove *wdb* on
    success. If merging raises an exception, *wdb* is preserved on disk."""
    from .store import Store

    ws = Store(wdb)
    try:
        tgt = Store(target_db)
        try:
            tgt.merge_from(ws, full_run=full_run)
        finally:
            tgt.close()
    finally:
        ws.close()
    try:
        os.remove(wdb)
    except OSError:
        pass


def _merge_dbs(target_db: str, worker_dbs: list[str], full_run: bool = True) -> None:
    """Fold each worker's tables into the target db, then remove merged temp dbs."""
    for wdb in worker_dbs:
        if os.path.exists(wdb):
            _merge_worker_db(target_db, wdb, full_run=full_run)


def _should_skip_existing(
    src: str,
    out_arg: str,
    image_only_arg: str | None,
    suffix: str,
    docs_with_pages: set[str],
) -> bool:
    """True iff the primary output PDF exists, the requested image-only PDF (if
    any) also exists, and *docs_with_pages* already contains this source's doc_id."""
    out_pdf = _resolve_out(out_arg, src, suffix, mkdir=False)
    if not os.path.exists(out_pdf):
        return False
    if image_only_arg:
        img_pdf = _resolve_out(image_only_arg, src, suffix, mkdir=False)
        if not os.path.exists(img_pdf):
            return False
    doc_id = os.path.basename(src)
    return doc_id in docs_with_pages


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hokusai-press",
        description="Scan PDF -> clean, light, searchable e-book PDF "
        "(non-destructive, batch-first).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="analyze + render one source to a searchable PDF")
    p_run.add_argument("source", nargs="+",
                       help="input PDF/image file(s), folder(s) (all *.pdf), or glob")
    p_run.add_argument("--out", required=True,
                       help="output PDF path, or a folder (saves <source>.pdf)")
    p_run.add_argument("--db", default="hokusai.db", help="job/decision SQLite db")
    p_run.add_argument("--model-dir", default="models", help="hybrid-ocr model dir")
    p_run.add_argument("--ocr-engine", default=DEFAULT_OCR_ENGINE,
                       choices=["auto", "hybrid", "yomitoku", "ndlocr",
                                "ppocr-v6", "paddleocr-v6",
                                "pp-structurev3", "pp-structure",
                                "paddle-vl", "paddleocr-vl",
                                "paddle-vl-1.6", "paddleocr-vl-1.6", "none"],
                       help="legacy combined engine selector. Prefer "
                            "--layout-engine and --text-engine for new runs")
    p_run.add_argument("--layout-engine", default=None,
                       choices=["none", "yomitoku", "pp-structurev3",
                                "pp-structure", "paddle-vl", "paddleocr-vl",
                                "paddle-vl-1.6", "paddleocr-vl-1.6"],
                       help="layout engine: Yomitoku RT-DETR, PP-StructureV3, "
                            "PaddleOCR-VL, or none")
    p_run.add_argument("--text-engine", default=None,
                       choices=["none", "yomitoku", "ndlocr", "ppocr-v6", "paddleocr-v6",
                                "paddle-vl", "paddleocr-vl",
                                "paddle-vl-1.6", "paddleocr-vl-1.6"],
                       help="text OCR engine: Yomitoku, NDL-OCR Lite, PP-OCRv6, "
                            "PaddleOCR-VL, or none")
    p_run.add_argument("--device", default="auto", choices=["auto", "cpu", "npu", "cuda", "dml", "qnn", "mps"],
                       help="compute device for OCR. hybrid: auto|cpu|npu|cuda|dml|qnn; "
                            "Paddle: cpu|gpu|npu[:0]|xpu[:0]|... as supported "
                            "by the installed Paddle runtime")
    p_run.add_argument("--runtime", default=None,
                       choices=["auto", "openvino", "onnxruntime", "paddle",
                                "transformers", "mlx"],
                       help="runtime/backend for selected engines")
    p_run.add_argument(
        "--mlx-model",
        default="huggingfinger0/PaddleOCR-VL-1.6-8bit",
        help="Hugging Face model ID served by mlx-vlm when --runtime mlx",
    )
    p_run.add_argument(
        "--mlx-server-url", default="http://127.0.0.1:8111/",
        help="mlx-vlm OpenAI-compatible server URL when --runtime mlx",
    )
    p_run.add_argument("--paddle-engine", default=None,
                       choices=["paddle", "transformers", "onnxruntime"],
                       help="deprecated alias for --runtime")
    p_run.add_argument("--no-ocr", action="store_true",
                       help="skip OCR (geometry + heuristic separation only)")
    p_run.add_argument("--keep-orientation", action="store_true",
                       help="leave pages that differ from the book's orientation as "
                            "stored (default: crop the scanner bed, then turn a page "
                            "saved sideways the way its text reads)")
    p_run.add_argument("--learned-model", default=None,
                       help="JSON page-kind model from `learn` to guide auto decisions")
    p_run.add_argument("--openvino-cache-dir", default=None,
                       help="OpenVINO compiled-model cache dir (speeds up NPU reuse)")
    p_run.add_argument("--profile", action="store_true",
                       help="print per-stage timing (raster/deskew/ocr/.../render)")
    p_run.add_argument("--workers", type=int, default=1,
                       help="process this many files in parallel threads sharing "
                            "one OCR engine (overlaps CPU work with NPU inference; "
                            "works with --device npu; ~2 is the sweet spot)")
    p_run.add_argument("--suffix", default="",
                       help="append this to the output file name when --out "
                            "(or --image-only-out) is a folder, e.g. '_HP' to "
                            "mark files made by HokusaiPress")
    p_run.add_argument("--image-only-out", default=None,
                       help="also write each result without its text layer "
                            "(images only) to this folder or file")
    p_run.add_argument("--skip-existing", action="store_true",
                       help="skip sources whose output PDF already exists "
                            "(resume an interrupted batch)")
    p_run.add_argument("--pages", default=None,
                       help="process only these pages, numbered as in the PDF "
                            "(1 = the first page), e.g. '10,53,237-238' "
                            "(fast for debugging)")
    p_run.add_argument("--no-nombre-align", action="store_true",
                       help="disable 3-pass folio-aligned margin refinement "
                            "(use Pass-1 type-area alignment only)")

    p_ocr = sub.add_parser(
        "ocr", help="replace the text layer of finished PDFs (images kept as is)")
    p_ocr.add_argument("source", nargs="+",
                       help="PDF file(s), folder(s) (all *.pdf), or glob")
    p_ocr.add_argument("--out", required=True,
                       help="output PDF path, or a folder (saves <source><suffix>.pdf)")
    p_ocr.add_argument("--suffix", default="",
                       help="append this to output names in a folder, e.g. '（OCR済）'")
    p_ocr.add_argument("--ocr-engine", default=DEFAULT_OCR_ENGINE,
                       help=f"OCR engine (default: {DEFAULT_OCR_ENGINE})")
    p_ocr.add_argument("--device", default="auto")
    p_ocr.add_argument("--model-dir", default="models")
    p_ocr.add_argument("--openvino-cache-dir", default=None)
    p_ocr.add_argument("--runtime", default=None)
    p_ocr.add_argument("--dpi", type=int, default=300,
                       help="render resolution for OCR (default 300)")
    p_ocr.add_argument("--keep-existing-text", action="store_true",
                       help="do not remove the text layer the PDF already has")
    p_ocr.add_argument("--text", action="store_true",
                       help="also write <output>.txt (pages separated by form feed)")
    p_ocr.add_argument("--skip-existing", action="store_true",
                       help="skip sources whose output PDF already exists")

    p_id = sub.add_parser(
        "identify", help="find each book's ISBN in its text layer, look it up "
                         "in NDL Search and propose a file name (nothing is renamed)")
    p_id.add_argument("source", nargs="+", help="OCR'd PDF file(s), folder(s), or glob")
    p_id.add_argument("--csv", required=True,
                      help="write proposals here (UTF-8 with BOM, opens in Excel); "
                           "edit new_name, then apply with `rename`")
    p_id.add_argument("--suffix", default="",
                      help="append this to every proposed name, e.g. '_HP（OCR済）'")
    p_id.add_argument("--keep-suffix-after", default=None,
                      help="keep the part of the current name after this marker, "
                           "e.g. '_HP' keeps '_HP（OCR済）'")

    p_rn = sub.add_parser("rename", help="rename files as listed in an identify CSV")
    p_rn.add_argument("--csv", required=True)
    p_rn.add_argument("--dry-run", action="store_true")

    p_queue = sub.add_parser("queue", help="list pages awaiting review")
    p_queue.add_argument("--db", default="hokusai.db")
    p_queue.add_argument("--doc", default=None, help="filter by doc id")

    p_web = sub.add_parser("review", help="start the review web UI")
    p_web.add_argument("--db", default="hokusai.db")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8765)

    p_re = sub.add_parser("rebuild",
                          help="regenerate a PDF from stored (corrected) params")
    p_re.add_argument("source", help="original source PDF/image")
    p_re.add_argument("--doc", required=True, help="doc id (source basename)")
    p_re.add_argument("--out", required=True,
                      help="output PDF path, or a folder (saves <source>.pdf)")
    p_re.add_argument("--db", default="hokusai.db")
    p_re.add_argument("--workers", type=int, default=1,
                      help="parallelize per-page render+encode across this "
                           "many threads (no OCR engine involved here, so "
                           "there's no NPU concurrency limit -- core count "
                           "is the natural ceiling)")
    p_re.add_argument("--margin-mode", choices=["normalize", "none"], default="normalize",
                      help="none = keep the whole scanned page (deskew and "
                           "binarize only; no crop, margin fill or shadow removal)")
    p_re.add_argument("--image-only-out", default=None,
                      help="also write the result without its text layer to this folder or file")
    p_re.add_argument("--suffix", default="",
                      help="append this to the output name when --out is a folder")
    p_re.add_argument("--process-pool", action="store_true",
                      help="use processes instead of threads for --workers "
                           "(faster: PIL's TIFF/group4 encode doesn't fully "
                           "release the GIL, capping thread speedup; "
                           "processes pay numpy-array pickling cost instead)")

    p_learn = sub.add_parser("learn",
                             help="train a page-kind model from the decision log")
    p_learn.add_argument("--db", default="hokusai.db")
    p_learn.add_argument("--out", default="hokusai_model.json")

    p_remargin = sub.add_parser(
        "remargin",
        help="re-run shadow/margin/content-box detection from STORED OCR "
             "regions (no OCR re-run) and rebuild the PDF -- for iterating "
             "on margin.py without re-paying OCR")
    p_remargin.add_argument("source", nargs="+",
                            help="original source PDF/image(s) the stored "
                                 "doc_id(s) were analyzed from")
    p_remargin.add_argument("--out", required=True,
                            help="output PDF path, or a folder (saves <source>.pdf)")
    p_remargin.add_argument("--db", default="hokusai.db")
    p_remargin.add_argument("--output-margin-mm", type=float, default=5.0)
    p_remargin.add_argument("--workers", type=int, default=1,
                            help="parallelize per-page margin-recompute and "
                                 "render+encode across this many threads "
                                 "(no OCR here, so core count is the ceiling)")
    p_remargin.add_argument("--process-pool", action="store_true",
                            help="use processes instead of threads for the "
                                 "render+encode half of --workers (see "
                                 "rebuild --process-pool)")
    p_remargin.add_argument("--pages", default=None,
                            help="only RENDER these pages (PDF page "
                                 "numbers, 1 = the first) to the output "
                                 "PDF, e.g. '1-30' (fast preview). "
                                 "Margin recompute still uses the WHOLE "
                                 "book's stored pages -- the robust per-side "
                                 "statistics need the full book to be "
                                 "representative; only the final render is "
                                 "limited.")
    p_remargin.add_argument("--no-nombre-align", action="store_true",
                            help="disable 3-pass folio-aligned margin refinement "
                                 "(use Pass-1 type-area alignment only)")

    p_cover = sub.add_parser(
        "cover",
        help="fetch a book cover (openBD / hanmoto.com / Google Books), upscale "
             "without margins, and prepend cover + blank verso page to PDFs")
    p_cover.add_argument("source", nargs="+",
                         help="PDF file(s), folder(s) (all *.pdf), or glob")
    p_cover.add_argument("--out", required=True,
                         help="output PDF path, or a folder (saves <source><suffix>.pdf)")
    p_cover.add_argument("--suffix", default="",
                         help="append this to output names in a folder")
    p_cover.add_argument("--db", default=None,
                         help="optional hokusai.db for image-only PDFs without a text layer")
    p_cover.add_argument("--isbn", default=None,
                         help="override ISBN instead of extracting from the PDF")
    p_cover.add_argument("--cover-image", default=None,
                         help="use a local cover image file instead of fetching online")
    p_cover.add_argument("--sources", default="openbd,hanmoto,google",
                         help="comma-separated cover lookup order "
                              "(default: openbd,hanmoto,google)")
    p_cover.add_argument("--cache-dir", default=None,
                         help="cache downloaded cover images by ISBN in this folder")
    p_cover.add_argument("--upscale", default="auto",
                         choices=["auto", "onnx", "lanczos", "none"],
                         help="upscale mode when fetched cover is smaller than body DPI "
                              "(default: auto)")
    p_cover.add_argument("--upscale-model", default=None,
                         help="optional ONNX super-resolution model path for upscale=auto/onnx")
    p_cover.add_argument("--target-dpi", type=int, default=300,
                         help="target effective DPI relative to body height (default: 300)")
    p_cover.add_argument("--jpeg-quality", type=int, default=94,
                         help="JPEG quality for embedded cover page (default: 94)")
    p_cover.add_argument("--no-blank-back", action="store_true",
                         help="do not insert a blank verso page after the cover")
    p_cover.add_argument("--replace-cover", action="store_true",
                         help="replace existing page 1 instead of prepending")
    p_cover.add_argument("--force", action="store_true",
                         help="insert even if /HokusaiPressCover metadata is already set")
    p_cover.add_argument("--skip-existing", action="store_true",
                         help="skip sources whose output PDF already exists")
    p_cover.add_argument("--dry-run", action="store_true",
                         help="resolve ISBN and fetch cover without writing output PDFs")
    p_cover.add_argument("--ocr-fallback", action="store_true",
                         help="OCR tail/head pages if PDF text layer and DB have no text")
    p_cover.add_argument("--ocr-engine", default=DEFAULT_OCR_ENGINE)
    p_cover.add_argument("--device", default="auto")
    p_cover.add_argument("--model-dir", default="models")
    p_cover.add_argument("--openvino-cache-dir", default=None)
    p_cover.add_argument("--runtime", default=None)

    args = parser.parse_args(argv)

    if args.command == "run":
        if (args.runtime or args.paddle_engine) == "mlx":
            os.environ["HOKUSAI_MLX_MODEL"] = args.mlx_model
            os.environ["HOKUSAI_MLX_SERVER_URL"] = args.mlx_server_url
        sources = _expand_sources(args.source)
        if not sources:
            print("no input files matched")
            return 1
        if len(sources) > 1 and not _is_folder_out(args.out):
            print("error: --out must be a folder when processing multiple inputs")
            return 1

        if (args.image_only_out and len(sources) > 1
                and not _is_folder_out(args.image_only_out)):
            print("error: --image-only-out must be a folder when processing multiple inputs")
            return 1
        planned = []
        for src in sources:
            planned.append((src, _resolve_out(args.out, src, args.suffix, mkdir=False)))
            if args.image_only_out:
                planned.append((
                    src,
                    _resolve_out(args.image_only_out, src, args.suffix, mkdir=False),
                ))
        if err := _check_output_collisions(sources, planned):
            print(err)
            return 1
        if err := _check_doc_id_collisions(sources, args.db):
            print(err)
            return 1
        if args.skip_existing:
            docs_with_pages: set[str] = set()
            if os.path.exists(args.db):
                from .store import Store

                st = Store(args.db)
                try:
                    docs_with_pages = set(st.doc_ids())
                finally:
                    st.close()
            kept = [
                s for s in sources
                if not _should_skip_existing(
                    s, args.out, args.image_only_out, args.suffix, docs_with_pages
                )
            ]
            for s in sources:
                if s not in kept:
                    print(f"[skip] {os.path.basename(s)}: output exists")
            sources = kept
            if not sources:
                return 0

        try:
            pages_sel = _parse_pages(args.pages)
        except ValueError as e:
            print(f"error: {e}")
            return 1

        def _payload(src, db):
            return {
                "src": src, "out": _resolve_out(args.out, src, args.suffix), "db": db,
                "image_only": (_resolve_out(args.image_only_out, src, args.suffix)
                               if args.image_only_out else None),
                "model_dir": args.model_dir, "device": args.device,
                "ocr_engine": args.ocr_engine,
                "layout_engine": args.layout_engine,
                "text_engine": args.text_engine,
                "use_ocr": not args.no_ocr, "learned_model": args.learned_model,
                "cache": args.openvino_cache_dir,
                "runtime": args.runtime or args.paddle_engine or "paddle",
                "pages": pages_sel,
                "keep_orientation": args.keep_orientation,
                "align_nombre": not args.no_nombre_align,
            }

        flagged_docs = 0
        failed = 0
        full_run = pages_sel is None
        parallel = args.workers > 1 and len(sources) > 1
        if parallel:
            # THREAD pool in ONE process (not multiprocessing): a single NPU/OCR
            # engine context is shared across threads, while each thread's CPU
            # work (raster/deskew/margin/MRC/render/recognition) overlaps another
            # thread's NPU inference (session.run + cv2/numpy release the GIL).
            # This makes --device npu safe in parallel -- unlike multiprocessing,
            # which opened one NPU context per process and hung the device.
            # Each thread writes its own temp db (no sqlite lock contention),
            # merged serially on the main thread into --db as each book succeeds.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            tmpdir = tempfile.mkdtemp(prefix="hokusai-")
            futures = {}
            with ThreadPoolExecutor(
                    max_workers=min(args.workers, len(sources))) as ex:
                for k, src in enumerate(sources):
                    wdb = os.path.join(tmpdir, f"w{k}.db")
                    fut = ex.submit(_run_one, _payload(src, wdb))
                    futures[fut] = (src, wdb)
                for fut in as_completed(futures):
                    src, wdb = futures[fut]
                    try:
                        summary = fut.result()
                    except Exception as e:
                        failed += 1
                        print(f"[FAIL] {os.path.basename(src)}: {e}", flush=True)
                        if os.path.exists(wdb):
                            try:
                                os.remove(wdb)
                            except OSError:
                                pass
                        continue
                    try:
                        _merge_worker_db(args.db, wdb, full_run=full_run)
                    except Exception as e:
                        failed += 1
                        print(
                            f"[FAIL] {os.path.basename(src)}: DB merge failed ({e}); "
                            f"temp DB kept at {wdb}",
                            flush=True,
                        )
                        continue
                    if _print_summary(summary, args.profile):
                        flagged_docs += 1
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass
        else:
            for src in sources:
                try:
                    summary = _run_one(_payload(src, args.db))
                except Exception as e:
                    failed += 1
                    print(f"[FAIL] {os.path.basename(src)}: {e}", flush=True)
                    continue
                if _print_summary(summary, args.profile):
                    flagged_docs += 1

        if len(sources) > 1:
            print(f"[done] {len(sources)} files, {flagged_docs} need review")
        if flagged_docs:
            print("  run: hokusai-press review")
        return 1 if failed else 0

    if args.command == "ocr":
        from .ocr import get_ocr_engine
        from .reocr import ocr_pdf

        sources = _expand_sources(args.source)
        if not sources:
            print("no input files matched")
            return 1
        if len(sources) > 1 and not _is_folder_out(args.out):
            print("error: --out must be a folder when processing multiple inputs")
            return 1
        planned = []
        for src in sources:
            dst = _resolve_out(args.out, src, args.suffix, mkdir=False)
            planned.append((src, dst))
            if args.text:
                planned.append((src, os.path.splitext(dst)[0] + ".txt"))
        if err := _check_output_collisions(sources, planned):
            print(err)
            return 1
        engine = get_ocr_engine(args.ocr_engine, args.model_dir, args.device,
                                args.openvino_cache_dir, args.runtime)
        failed = 0
        for src in sources:
            dst = _resolve_out(args.out, src, args.suffix)
            if args.skip_existing and os.path.exists(dst):
                print(f"[skip] {os.path.basename(src)}: output exists")
                continue
            try:
                r = ocr_pdf(src, dst, engine, dpi=args.dpi,
                            keep_existing_text=args.keep_existing_text,
                            text_out=(os.path.splitext(dst)[0] + ".txt") if args.text else None)
            except Exception as e:  # keep going with the other books
                failed += 1
                print(f"[FAIL] {os.path.basename(src)}: {e}")
                continue
            print(f"[OK] {os.path.basename(src)}: {r['pages']} pages, {r['lines']} lines, "
                  f"{r['chars']} chars -> {dst}  ({r['seconds'] / max(r['pages'], 1):.2f}s/page)",
                  flush=True)
        return 1 if failed else 0

    if args.command == "identify":
        import csv

        from .model import ensure_parent_dir
        from .naming import identify

        sources = _expand_sources(args.source)
        if not sources:
            print("no input files matched")
            return 1
        ensure_parent_dir(args.csv)
        rows = []
        for src in sources:
            suffix = args.suffix
            if args.keep_suffix_after:
                stem = os.path.splitext(os.path.basename(src))[0]
                k = stem.find(args.keep_suffix_after)
                if k >= 0:
                    suffix = stem[k:] + suffix
            try:
                p = identify(src, suffix=suffix)
            except Exception as e:
                print(f"[FAIL] {os.path.basename(src)}: {e}")
                rows.append({"path": src, "new_name": "", "note": f"error: {e}"})
                continue
            r = p.record
            rows.append({
                "path": src, "new_name": p.name or "",
                "isbn": r.isbn if r else "", "title": r.title if r else "",
                "authors": r.authors if r else "", "publisher": r.publisher if r else "",
                "issued": r.issued if r else "", "score": p.score, "note": p.note,
                "isbns_found": " ".join(p.isbn_pages),
            })
            mark = "OK" if p.name and not p.note else "CHECK"
            print(f"[{mark}] {os.path.basename(src)} -> {p.name or '-'}  {p.note}", flush=True)
        fields = ["path", "new_name", "isbn", "title", "authors", "publisher",
                  "issued", "score", "note", "isbns_found"]
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fields})
        print(f"proposals -> {args.csv}  (check new_name, then: hokusai-press rename --csv ...)")
        return 0

    if args.command == "rename":
        import csv

        with open(args.csv, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        bad = 0
        for row in rows:
            src, name = row.get("path", ""), (row.get("new_name") or "").strip()
            if not name:
                continue
            dst = os.path.join(os.path.dirname(src), name)
            if os.path.abspath(dst) == os.path.abspath(src):
                continue
            if not os.path.exists(src):
                print(f"[skip] missing: {src}")
                bad += 1
                continue
            if os.path.exists(dst):
                print(f"[skip] already exists: {dst}")
                bad += 1
                continue
            print(f"{'[dry-run] ' if args.dry_run else ''}{os.path.basename(src)} -> {name}")
            if not args.dry_run:
                # a sync client (OneDrive) may hold the file for a moment
                for attempt in range(5):
                    try:
                        os.rename(src, dst)
                        break
                    except PermissionError as e:
                        if attempt == 4:
                            print(f"[skip] in use: {src} ({e})")
                            bad += 1
                        else:
                            import time

                            time.sleep(3)
        return 1 if bad else 0

    if args.command == "queue":
        from .store import Store

        store = Store(args.db)
        try:
            rows = store.review_queue(args.doc)
        finally:
            store.close()
        if not rows:
            print("review queue is empty")
            return 0
        for r in rows:
            print(f"{r.doc_id} p{r.page_index + 1}: {', '.join(r.flags)}")  # PDF page
        return 0

    if args.command == "review":
        from .webui.app import serve

        serve(args.db, host=args.host, port=args.port)
        return 0

    if args.command == "rebuild":
        from .pipeline import rebuild

        planned = [(args.source, _resolve_out(args.out, args.source, args.suffix, mkdir=False))]
        if args.image_only_out:
            planned.append((
                args.source,
                _resolve_out(args.image_only_out, args.source, args.suffix, mkdir=False),
            ))
        if err := _check_output_collisions([args.source], planned):
            print(err)
            return 1
        out = _resolve_out(args.out, args.source, args.suffix)
        bare = (_resolve_out(args.image_only_out, args.source, args.suffix)
                if args.image_only_out else None)
        try:
            summary = rebuild(args.doc, args.source, out, db_path=args.db,
                              max_workers=args.workers,
                              use_processes=args.process_pool,
                              margin_mode=args.margin_mode)
        except ValueError as e:
            print(f"error: {e}")
            return 1
        print(f"[OK] rebuilt {summary['doc_id']}: {summary['pages']} pages "
              f"-> {summary['out_pdf']}", flush=True)
        if bare:
            from .textlayer import image_only_copy

            image_only_copy(out, bare)
            print(f"  [image-only] {bare}", flush=True)
        return 0

    if args.command == "remargin":
        from .pipeline import recompute_shadows_and_margins, rebuild
        from .store import Store

        sources = _expand_sources(args.source)
        if not sources:
            print("no input files matched")
            return 1
        if len(sources) > 1 and not _is_folder_out(args.out):
            print("error: --out must be a folder when processing multiple inputs")
            return 1
        planned = [(src, _resolve_out(args.out, src, mkdir=False)) for src in sources]
        if err := _check_output_collisions(sources, planned):
            print(err)
            return 1
        if err := _check_doc_id_collisions(sources, args.db):
            print(err)
            return 1

        try:
            pages_sel = _parse_pages(args.pages)
        except ValueError as e:
            print(f"error: {e}")
            return 1
        store = Store(args.db)
        try:
            for src in sources:
                doc_id = os.path.basename(src)
                out = _resolve_out(args.out, src)
                try:
                    n = recompute_shadows_and_margins(
                        store, doc_id, src, output_margin_mm=args.output_margin_mm,
                        max_workers=args.workers,
                        align_nombre=not args.no_nombre_align)
                    summary = rebuild(doc_id, src, out, db_path=args.db,
                                      max_workers=args.workers,
                                      use_processes=args.process_pool,
                                      pages=pages_sel)
                except ValueError as e:
                    print(f"error: {e}")
                    return 1
                print(f"[OK] remargin {doc_id}: {n} pages recomputed (no OCR), "
                      f"{summary['pages']} rendered -> {summary['out_pdf']}")
        finally:
            store.close()
        return 0

    if args.command == "learn":
        from . import learn
        from .store import Store

        store = Store(args.db)
        try:
            decisions = store.decisions_for_training("page_kind")
        finally:
            store.close()
        model = learn.train(decisions)
        if model is None:
            print(f"not enough labeled decisions yet "
                  f"(need >= {learn.MIN_PER_CLASS} per class, >= 2 classes)")
            return 1
        learn.save(model, args.out)
        print(f"[OK] trained page-kind model -> {args.out} "
              f"(classes: {model['counts']})")
        return 0

    if args.command == "cover":
        import cv2

        from .cover import (
            extract_isbn,
            fetch_cover_image,
            insert_cover_pages,
            isbn10_to_13,
        )

        sources = _expand_sources(args.source)
        if not sources:
            print("no input files matched")
            return 1
        if len(sources) > 1 and not _is_folder_out(args.out):
            print("error: --out must be a folder when processing multiple inputs")
            return 1
        if len(sources) > 1 and args.isbn:
            print("error: --isbn cannot be used with multiple input files")
            return 1
        if not args.dry_run:
            planned = [
                (src, _resolve_out(args.out, src, args.suffix, mkdir=False))
                for src in sources
            ]
            if err := _check_output_collisions(sources, planned):
                print(err)
                return 1

        fetch_sources = tuple(
            s.strip() for s in args.sources.split(",") if s.strip()
        )
        local_cover = None
        if args.cover_image:
            local_cover = cv2.imread(args.cover_image, cv2.IMREAD_COLOR)
            if local_cover is None:
                print(f"error: failed to read cover image: {args.cover_image}")
                return 1

        ocr_engine = None
        if args.ocr_fallback:
            from .ocr import get_ocr_engine

            ocr_engine = get_ocr_engine(
                args.ocr_engine, args.model_dir, args.device,
                args.openvino_cache_dir, args.runtime,
            )

        failed = 0
        for src in sources:
            dst = _resolve_out(args.out, src, args.suffix, mkdir=not args.dry_run)
            if not args.dry_run and args.skip_existing and os.path.exists(dst):
                print(f"[skip] {os.path.basename(src)}: output exists")
                continue
            try:
                if local_cover is not None:
                    cover_bgr = local_cover
                    isbn13 = isbn10_to_13(args.isbn) if args.isbn else None
                    cover_src = "local"
                else:
                    if args.isbn:
                        isbn13 = isbn10_to_13(args.isbn)
                        isbn_note = "cli"
                    else:
                        isbn13, isbn_note = extract_isbn(
                            src, db_path=args.db, ocr_engine=ocr_engine,
                        )
                    if not isbn13:
                        print(f"[skip] {os.path.basename(src)}: {isbn_note}")
                        failed += 1
                        continue
                    fetched = fetch_cover_image(
                        isbn13,
                        sources=fetch_sources,
                        cache_dir=args.cache_dir,
                        throttle_sec=0.3,
                    )
                    if fetched is None:
                        print(f"[skip] {os.path.basename(src)}: no cover found for ISBN {isbn13}")
                        failed += 1
                        continue
                    cover_bgr = fetched.image_bgr
                    cover_src = fetched.source

                if args.dry_run:
                    h, w = cover_bgr.shape[:2]
                    print(
                        f"[dry-run] {os.path.basename(src)}: ISBN={isbn13 or '-'} "
                        f"source={cover_src} size={w}x{h} -> {dst}",
                        flush=True,
                    )
                    continue

                res = insert_cover_pages(
                    src,
                    dst,
                    cover_bgr,
                    isbn=isbn13,
                    source_name=cover_src,
                    blank_back=not args.no_blank_back,
                    replace_cover=args.replace_cover,
                    force=args.force,
                    upscale_mode=args.upscale,
                    upscale_model=args.upscale_model,
                    target_dpi=args.target_dpi,
                    jpeg_quality=args.jpeg_quality,
                )
                if res.skipped:
                    print(f"[skip] {os.path.basename(src)}: {res.note}", flush=True)
                else:
                    print(
                        f"[OK] {os.path.basename(src)}: ISBN={res.isbn or '-'} "
                        f"src={res.source} upscale={res.upscale_method} "
                        f"({res.cover_size_px[0]}x{res.cover_size_px[1]} -> "
                        f"{res.cover_box_pt[0]:.1f}x{res.cover_box_pt[1]:.1f}pt) -> {dst}",
                        flush=True,
                    )
            except Exception as e:
                failed += 1
                print(f"[FAIL] {os.path.basename(src)}: {e}", flush=True)
        return 1 if failed else 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
