#!/usr/bin/env python3
"""Overlay final folio decisions on an original PDF without rasterizing it."""

from __future__ import annotations

import argparse
import io
import json
import math
import sqlite3
from pathlib import Path

import pikepdf
from reportlab.pdfgen import canvas


COLORS = {
    "main_ocr": (0.85, 0.0, 0.75),       # magenta
    "dedicated_ocr": (0.0, 0.65, 0.85),  # cyan
    "both": (0.15, 0.2, 0.95),           # blue
    "series": (1.0, 0.45, 0.0),          # orange
    "none": (0.4, 0.4, 0.4),             # gray
}


def _inverse_deskew(x, y, width, height, angle_deg):
    angle = math.radians(-angle_deg)
    cx, cy = width / 2.0, height / 2.0
    dx, dy = x - cx, y - cy
    return (cx + math.cos(angle) * dx - math.sin(angle) * dy,
            cy + math.sin(angle) * dx + math.cos(angle) * dy)


def _pdf_points(box, params, pdf_width, pdf_height):
    margin = params.get("margin") or {}
    width = float(margin.get("page_w") or 1.0)
    height = float(margin.get("page_h") or 1.0)
    angle = float((params.get("deskew") or {}).get("angle_deg") or 0.0)
    corners = [(box["x0"], box["y0"]), (box["x1"], box["y0"]),
               (box["x1"], box["y1"]), (box["x0"], box["y1"])]
    result = []
    for x, y in corners:
        ox, oy = _inverse_deskew(float(x), float(y), width, height, angle)
        result.append((ox / width * pdf_width,
                       pdf_height - oy / height * pdf_height))
    return result


def _channel(decision):
    channels = set(decision.get("support_channels") or [])
    if channels == {"main_ocr", "dedicated_ocr"}:
        return "both", "BOTH"
    if "dedicated_ocr" in channels:
        return "dedicated_ocr", "DEDICATED"
    if "main_ocr" in channels:
        return "main_ocr", "MAIN"
    if decision.get("value") is not None:
        return "series", "SERIES"
    return "none", "NONE"


def _overlay_page(params, pdf_width, pdf_height):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(pdf_width, pdf_height), pageCompression=1)
    decision = params.get("nombre_decision") or {}
    channel, label = _channel(decision)
    color = COLORS[channel]
    text = decision.get("text")
    value = decision.get("value")
    shown = text if text is not None else (str(value) if value is not None else "UNNUMBERED")
    title = f"DECIDED: {shown} [{label}]"
    box = decision.get("box")
    c.setStrokeColorRGB(*color)
    c.setFillColorRGB(*color)
    c.setLineWidth(1.8)
    if box:
        points = _pdf_points(box, params, pdf_width, pdf_height)
        path = c.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        path.close()
        c.drawPath(path, stroke=1, fill=0)
        tx = min(max(8.0, min(x for x, _ in points)), max(8.0, pdf_width - 145.0))
        ty = max(10.0, min(y for _, y in points) - 13.0)
    else:
        tx, ty = 8.0, pdf_height - 18.0
    c.setFont("Helvetica-Bold", 8)
    text_width = c.stringWidth(title, "Helvetica-Bold", 8)
    c.setFillColorRGB(1.0, 1.0, 0.55)
    c.rect(tx - 2, ty - 2, text_width + 4, 11, stroke=0, fill=1)
    c.setStrokeColorRGB(*color)
    c.rect(tx - 2, ty - 2, text_width + 4, 11, stroke=1, fill=0)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(tx, ty, title)
    c.save()
    buf.seek(0)
    return buf


def export(db_path: str, source: str, out_path: str) -> None:
    doc_id = Path(source).name
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT page_index, params_json FROM pages WHERE doc_id=? ORDER BY page_index",
        (doc_id,),
    ).fetchall()
    con.close()
    if not rows:
        raise SystemExit(f"no pages in {db_path}: {doc_id}")
    params_by_index = {int(i): json.loads(raw) for i, raw in rows}

    with pikepdf.Pdf.open(source) as pdf:
        for index, page in enumerate(pdf.pages):
            params = params_by_index.get(index)
            if params is None:
                continue
            media = page.MediaBox
            width = float(media[2]) - float(media[0])
            height = float(media[3]) - float(media[1])
            with pikepdf.Pdf.open(_overlay_page(params, width, height)) as overlay:
                page.add_overlay(overlay.pages[0])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pdf.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    export(args.db, args.source, args.out)


if __name__ == "__main__":
    main()
