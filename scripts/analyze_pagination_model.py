#!/usr/bin/env python3
"""Reclassify stored folio results into observed/logical/navigation layers."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from hokusai_press import nombre
from hokusai_press.model import _decode_page


def _format_label_runs(pages) -> str:
    """Compress labels only while their numeric value advances by exactly one."""
    items = []
    for index, page in enumerate(pages, start=1):
        decision = page.pagination
        if decision.logical_number is None:
            items.append(("scan", index, decision.pdf_label or f"scan-{index}"))
        else:
            family = decision.numbering_system.value
            items.append((family, decision.logical_number,
                          decision.pdf_label or str(decision.logical_number)))
    if not items:
        return "-"
    runs = []
    start = previous = items[0]
    for item in items[1:]:
        if item[0] == previous[0] and item[1] == previous[1] + 1:
            previous = item
            continue
        runs.append(start[2] if start == previous else f"{start[2]}-{previous[2]}")
        start = previous = item
    runs.append(start[2] if start == previous else f"{start[2]}-{previous[2]}")
    return ",".join(runs)


def analyze(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    docs = [row[0] for row in con.execute(
        "SELECT DISTINCT doc_id FROM pages ORDER BY doc_id")]
    result = {"documents": []}
    totals = Counter()
    for doc_id in docs:
        pages = [_decode_page(json.loads(row[0])) for row in con.execute(
            "SELECT params_json FROM pages WHERE doc_id=? ORDER BY page_index",
            (doc_id,),
        )]
        before_numbered = sum(p.page_number is not None for p in pages)
        nombre.finalize_pagination(pages)
        roles = Counter(p.pagination.role.value for p in pages)
        methods = Counter(p.pagination.method.value for p in pages)
        conflicts = [p.source.page_index + 1 for p in pages
                     if p.pagination.conflict]
        uncounted = [p.source.page_index + 1 for p in pages
                     if p.pagination.role.value == "uncounted"]
        row = {
            "doc_id": doc_id,
            "pages": len(pages),
            "before_numbered": before_numbered,
            "after_logical": sum(p.pagination.logical_number is not None
                                 for p in pages),
            "roles": dict(roles),
            "methods": dict(methods),
            "conflict_pages": conflicts,
            "uncounted_pages": uncounted,
            "pdf_page_labels": _format_label_runs(pages),
        }
        result["documents"].append(row)
        totals.update(roles)
        totals["conflicts"] += len(conflicts)
        totals["pages"] += len(pages)
    con.close()
    result["totals"] = dict(totals)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--out")
    parser.add_argument("--text-out")
    args = parser.parse_args()
    result = analyze(args.db)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if args.text_out:
        lines = [
            f"{row['doc_id']} [PDF PageLabels]: {row['pdf_page_labels']}"
            for row in result["documents"]
        ]
        Path(args.text_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
