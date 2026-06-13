"""HokusaiPress CLI."""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hokusai-press",
        description="Scan PDF -> clean, light, searchable e-book PDF "
        "(non-destructive, batch-first).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="analyze + render one source to a searchable PDF")
    p_run.add_argument("source", help="input PDF or image")
    p_run.add_argument("--out", required=True, help="output PDF path")
    p_run.add_argument("--db", default="hokusai.db", help="job/decision SQLite db")
    p_run.add_argument("--model-dir", default="models", help="hybrid-ocr model dir")
    p_run.add_argument("--device", default="auto",
                       help="OCR device: auto|cpu|npu|cuda|dml|qnn")
    p_run.add_argument("--no-ocr", action="store_true",
                       help="skip OCR (geometry + heuristic separation only)")

    p_queue = sub.add_parser("queue", help="list pages awaiting review")
    p_queue.add_argument("--db", default="hokusai.db")
    p_queue.add_argument("--doc", default=None, help="filter by doc id")

    p_web = sub.add_parser("review", help="start the review web UI")
    p_web.add_argument("--db", default="hokusai.db")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)

    if args.command == "run":
        from .pipeline import run

        summary = run(
            args.source, args.out, db_path=args.db, model_dir=args.model_dir,
            device=args.device, use_ocr=not args.no_ocr,
        )
        print(f"[OK] {summary['doc_id']}: {summary['pages']} pages -> {summary['out_pdf']}")
        if summary["needs_review"]:
            print(f"  needs review: {len(summary['needs_review'])} pages "
                  f"{summary['needs_review']}")
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

    return 1


if __name__ == "__main__":
    sys.exit(main())
