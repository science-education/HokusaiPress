"""HokusaiPress CLI."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import time


def _is_folder_out(out: str) -> bool:
    """--out denotes a folder when it is an existing directory, ends with a path
    separator, or has no extension."""
    return (os.path.isdir(out) or out.endswith(("/", "\\"))
            or os.path.splitext(out)[1] == "")


def _resolve_out(out: str, source: str) -> str:
    """Allow --out to be a folder: save as the source's name with a .pdf
    extension inside it (folder created on demand). Otherwise --out is used as
    the output file path verbatim."""
    if _is_folder_out(out):
        os.makedirs(out, exist_ok=True)
        stem = os.path.splitext(os.path.basename(source))[0]
        return os.path.join(out, stem + ".pdf")
    return out


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
    """Parse a 0-based page selection like '9', '52,236-237', '0-4' to a set.
    Returns None for the whole document."""
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out or None


def _run_one(payload: dict) -> dict:
    """Process a single source to its own db (picklable; used by workers and the
    serial path). Returns the run summary plus timing and the db it wrote."""
    from .pipeline import run

    t0 = time.perf_counter()
    summary = run(
        payload["src"], payload["out"], db_path=payload["db"],
        model_dir=payload["model_dir"], device=payload["device"],
        ocr_engine=payload["ocr_engine"], use_ocr=payload["use_ocr"],
        learned_model_path=payload["learned_model"],
        openvino_cache_dir=payload["cache"],
        paddle_engine=payload["paddle_engine"], pages=payload["pages"],
    )
    summary["tpb"] = time.perf_counter() - t0
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
    print(line)
    for w in summary.get("warnings", []):
        print(f"  [warn] {w}")
    if profile:
        prof = summary.get("profile", {})
        total = sum(prof.values()) or 1.0
        for stage, sec in sorted(prof.items(), key=lambda kv: -kv[1]):
            print(f"  [prof] {stage:<16} {sec:7.1f}s "
                  f"{sec / pages:6.3f}s/page  {sec / total * 100:4.0f}%")
    return flagged


def _merge_dbs(target_db: str, worker_dbs: list[str]) -> None:
    """Fold each worker's pages into the target db, then remove the temp dbs."""
    from .store import Store

    tgt = Store(target_db)
    try:
        for wdb in worker_dbs:
            ws = Store(wdb)
            try:
                for doc in ws.doc_ids():
                    for row in ws.list_pages(doc):
                        tgt.upsert_page(doc, row.page_index, row.params)
            finally:
                ws.close()
    finally:
        tgt.close()
    for wdb in worker_dbs:
        try:
            os.remove(wdb)
        except OSError:
            pass


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
    p_run.add_argument("--ocr-engine", default="hybrid",
                       choices=["auto", "hybrid", "yomitoku", "ndlocr",
                                "ppocr-v6", "paddleocr-v6",
                                "paddle-vl", "paddleocr-vl"],
                       help="OCR engine: hybrid/Yomitoku-NDLOCR, PP-OCRv6, or "
                            "PaddleOCR-VL-1.6 document parser")
    p_run.add_argument("--device", default="auto",
                       help="OCR device. hybrid: auto|cpu|npu|cuda|dml|qnn; "
                            "Paddle: cpu|gpu|npu[:0]|xpu[:0]|... as supported "
                            "by the installed Paddle runtime")
    p_run.add_argument("--paddle-engine", default="paddle",
                       choices=["paddle", "transformers"],
                       help="PaddleOCR backend for ppocr-v6/paddle-vl")
    p_run.add_argument("--no-ocr", action="store_true",
                       help="skip OCR (geometry + heuristic separation only)")
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
    p_run.add_argument("--pages", default=None,
                       help="process only these 0-based pages, e.g. '9,52,236-237'"
                            " (fast for debugging)")

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

    p_learn = sub.add_parser("learn",
                             help="train a page-kind model from the decision log")
    p_learn.add_argument("--db", default="hokusai.db")
    p_learn.add_argument("--out", default="hokusai_model.json")

    args = parser.parse_args(argv)

    if args.command == "run":
        sources = _expand_sources(args.source)
        if not sources:
            print("no input files matched")
            return 1
        if len(sources) > 1 and not _is_folder_out(args.out):
            print("error: --out must be a folder when processing multiple inputs")
            return 1

        pages_sel = _parse_pages(args.pages)

        def _payload(src, db):
            return {
                "src": src, "out": _resolve_out(args.out, src), "db": db,
                "model_dir": args.model_dir, "device": args.device,
                "ocr_engine": args.ocr_engine,
                "use_ocr": not args.no_ocr, "learned_model": args.learned_model,
                "cache": args.openvino_cache_dir,
                "paddle_engine": args.paddle_engine, "pages": pages_sel,
            }

        flagged_docs = 0
        parallel = args.workers > 1 and len(sources) > 1
        if parallel:
            # THREAD pool in ONE process (not multiprocessing): a single NPU/OCR
            # engine context is shared across threads, while each thread's CPU
            # work (raster/deskew/margin/MRC/render/recognition) overlaps another
            # thread's NPU inference (session.run + cv2/numpy release the GIL).
            # This makes --device npu safe in parallel -- unlike multiprocessing,
            # which opened one NPU context per process and hung the device.
            # Each thread writes its own temp db (no sqlite lock contention),
            # merged into --db at the end.
            from concurrent.futures import ThreadPoolExecutor

            tmpdir = tempfile.mkdtemp(prefix="hokusai-")
            payloads, worker_dbs = [], []
            for k, src in enumerate(sources):
                wdb = os.path.join(tmpdir, f"w{k}.db")
                worker_dbs.append(wdb)
                payloads.append(_payload(src, wdb))
            with ThreadPoolExecutor(
                    max_workers=min(args.workers, len(sources))) as ex:
                for summary in ex.map(_run_one, payloads):
                    if _print_summary(summary, args.profile):
                        flagged_docs += 1
            _merge_dbs(args.db, worker_dbs)
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass
        else:
            for src in sources:
                summary = _run_one(_payload(src, args.db))
                if _print_summary(summary, args.profile):
                    flagged_docs += 1

        if len(sources) > 1:
            print(f"[done] {len(sources)} files, {flagged_docs} need review")
        if flagged_docs:
            print("  run: hokusai-press review")
        return 0

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
            print(f"{r.doc_id} p{r.page_index}: {', '.join(r.flags)}")
        return 0

    if args.command == "review":
        from .webui.app import serve

        serve(args.db, host=args.host, port=args.port)
        return 0

    if args.command == "rebuild":
        from .pipeline import rebuild

        out = _resolve_out(args.out, args.source)
        summary = rebuild(args.doc, args.source, out, db_path=args.db)
        print(f"[OK] rebuilt {summary['doc_id']}: {summary['pages']} pages "
              f"-> {summary['out_pdf']}")
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

    return 1


if __name__ == "__main__":
    sys.exit(main())
