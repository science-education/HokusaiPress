"""HokusaiPress CLI."""

from __future__ import annotations

import argparse
import glob
import os
import sys
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
    p_run.add_argument("--device", default="auto",
                       help="OCR device: auto|cpu|npu|cuda|dml|qnn")
    p_run.add_argument("--no-ocr", action="store_true",
                       help="skip OCR (geometry + heuristic separation only)")
    p_run.add_argument("--learned-model", default=None,
                       help="JSON page-kind model from `learn` to guide auto decisions")
    p_run.add_argument("--openvino-cache-dir", default=None,
                       help="OpenVINO compiled-model cache dir (speeds up NPU reuse)")

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
        from .pipeline import run

        sources = _expand_sources(args.source)
        if not sources:
            print("no input files matched")
            return 1
        if len(sources) > 1 and not _is_folder_out(args.out):
            print("error: --out must be a folder when processing multiple inputs")
            return 1

        flagged_docs = 0
        for src in sources:
            out = _resolve_out(args.out, src)
            t0 = time.perf_counter()
            summary = run(
                src, out, db_path=args.db, model_dir=args.model_dir,
                device=args.device, use_ocr=not args.no_ocr,
                learned_model_path=args.learned_model,
                openvino_cache_dir=args.openvino_cache_dir,
            )
            tpb = time.perf_counter() - t0
            pages = summary["pages"] or 1
            line = (f"[OK] {summary['doc_id']}: {summary['pages']} pages "
                    f"-> {summary['out_pdf']}")
            if summary["needs_review"]:
                flagged_docs += 1
                line += f"  (needs review: {len(summary['needs_review'])} pages)"
            line += f"  TPB={tpb:.1f}s TPP={tpb / pages:.2f}s/page"
            print(line)
            for w in summary.get("warnings", []):
                print(f"  [warn] {w}")
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
