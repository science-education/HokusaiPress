#!/usr/bin/env python3
"""Compare main OCR, dedicated folio OCR, and combined folio resolution."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sqlite3
from pathlib import Path
from typing import Iterable

from hokusai_press import nombre
from hokusai_press.model import Box, Region, RegionKind, _decode_page


def _iou(a, b) -> float:
    if not a or not b:
        return 0.0
    x0, y0 = max(a["x0"], b["x0"]), max(a["y0"], b["y0"])
    x1, y1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    aa = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
    bb = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
    return inter / max(1.0, aa + bb - inter)


def _format_page_runs(pages: Iterable[int]) -> str:
    runs = []
    start = prev = None
    for page in pages:
        if start is None:
            start = prev = page
            continue
        if page == prev + 1:
            prev = page
            continue
        runs.append((start, prev))
        start = prev = page
    if start is not None:
        runs.append((start, prev))
    parts = []
    for a, b in runs:
        parts.append(str(a) if a == b else f"{a}-{b}")
    return ",".join(parts)


def _dedicated_candidates(page):
    return [
        (item["band"], item["kind"], item["value"],
         Region(kind=RegionKind.TEXT, box=Box(**item["box"]),
                source="nombre_reader", ocr_text=item.get("text"),
                ocr_conf=item.get("confidence")))
        for item in page.nombre_evidence.get("dedicated_ocr", [])
    ]


def _ablate(reference, mode):
    pages = copy.deepcopy(reference)
    for page in pages:
        page.page_number = page.nombre_text = None
        page.nombre_decision = {}
        if page.margin:
            page.margin.nombre_box = None
        if mode == "dedicated":
            h = page.margin.page_h if page.margin else 0
            for region in page.regions:
                cy = (region.box.y0 + region.box.y1) / 2.0
                if h and (cy <= h * nombre.BAND_FRAC or cy >= h * (1 - nombre.BAND_FRAC)):
                    region.ocr_text = None
            page._nombre_candidates = _dedicated_candidates(page)
        else:
            page._nombre_candidates = []
    heights = [p.margin.page_h for p in pages]
    widths = [p.margin.page_w for p in pages]
    nombre.resolve(pages, heights, widths)
    return pages


def analyze(db_path):
    con = sqlite3.connect(db_path)
    docs = [x[0] for x in con.execute("select distinct doc_id from pages order by doc_id")]
    rows = []
    totals = {}
    for doc in docs:
        reference = [_decode_page(json.loads(x[0])) for x in con.execute(
            "select params_json from pages where doc_id=? order by page_index", (doc,))]
        variants = {"main": _ablate(reference, "main"),
                    "dedicated": _ablate(reference, "dedicated")}
        ref_numbered = sum(p.page_number is not None for p in reference)
        ref_boxed = sum(bool(p.nombre_decision.get("box")) for p in reference)
        reference_pages = _format_page_runs(
            p.source.page_index + 1 for p in reference if p.page_number is not None
        )
        for mode, pages in variants.items():
            value_match = box_match = joint = false_positive = 0
            diff_pages: list[int] = []
            for ref, got in zip(reference, pages):
                ref_dec = ref.nombre_decision
                ref_box = ref_dec.get("box")
                got_box = None if not got.margin or not got.margin.nombre_box else {
                    "x0": got.margin.nombre_box.x0, "y0": got.margin.nombre_box.y0,
                    "x1": got.margin.nombre_box.x1, "y1": got.margin.nombre_box.y1,
                }
                value_ok = ref.page_number is not None and got.page_number == ref.page_number
                box_ok = ref_box is not None and _iou(ref_box, got_box) >= 0.45
                value_match += value_ok
                box_match += box_ok
                joint += value_ok and box_ok
                false_positive += ref.page_number is None and got.page_number is not None
                if not (value_ok and box_ok):
                    diff_pages.append(ref.source.page_index + 1)
            rows.append({
                "doc_id": doc, "mode": mode, "reference_numbered": ref_numbered,
                "reference_boxed": ref_boxed, "value_match": value_match,
                "box_match": box_match, "value_and_box_match": joint,
                "false_positive_on_reference_unnumbered": false_positive,
                "diff_pages": _format_page_runs(diff_pages),
                "reference_pages": reference_pages,
            })
    con.close()
    for mode in ("main", "dedicated"):
        selected = [r for r in rows if r["mode"] == mode]
        totals[mode] = {key: sum(r[key] for r in selected) for key in (
            "reference_numbered", "reference_boxed", "value_match", "box_match",
            "value_and_box_match", "false_positive_on_reference_unnumbered")}
    return rows, totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows, totals = analyze(args.db)
    out = Path(args.out)
    out.with_suffix(".json").write_text(
        json.dumps({"per_document": rows, "totals": totals}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    lines = []
    for row in rows:
        if row["mode"] == "main":
            lines.append(
                f"{row['doc_id']} [reference]: {row['reference_pages'] or '-'}"
            )
        lines.append(
            f"{row['doc_id']} [{row['mode']}]: {row['diff_pages'] or '-'}"
        )
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
