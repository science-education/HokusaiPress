"""Structured text exports derived only from persisted PageParams."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .model import Document, RegionKind


def document_to_data(document: Document) -> dict[str, Any]:
    """Return the stable JSON-compatible representation used by all exports."""
    pages = []
    for output_index, page in enumerate(document.pages):
        elements = []
        for order, region in enumerate(page.regions):
            element: dict[str, Any] = {
                "order": order,
                "type": region.kind.value,
                "bbox": [
                    float(region.box.x0),
                    float(region.box.y0),
                    float(region.box.x1),
                    float(region.box.y1),
                ],
                "source": region.source,
            }
            if region.ocr_text:
                element["text"] = region.ocr_text
            if region.ocr_conf is not None:
                element["confidence"] = float(region.ocr_conf)
            if region.layout_label:
                element["layout_label"] = region.layout_label
            if region.tone:
                element["tone"] = region.tone
            elements.append(element)
        pages.append({
            "output_index": output_index,
            "source_page_index": page.source.page_index,
            "page_number": page.page_number,
            "nombre_text": page.nombre_text,
            "printed_folio": (None if page.printed_folio is None else {
                "text": page.printed_folio.text,
                "parsed_value": page.printed_folio.parsed_value,
                "numbering_system": page.printed_folio.numbering_system.value,
                "bbox": [page.printed_folio.box.x0, page.printed_folio.box.y0,
                         page.printed_folio.box.x1, page.printed_folio.box.y1],
                "confidence": page.printed_folio.confidence,
                "source": page.printed_folio.source,
            }),
            "pagination": {
                "logical_number": page.pagination.logical_number,
                "numbering_system": page.pagination.numbering_system.value,
                "role": page.pagination.role.value,
                "method": page.pagination.method.value,
                "confidence": page.pagination.confidence,
                "supporting_pages": page.pagination.supporting_pages,
                "conflict": page.pagination.conflict,
                "predicted_number": page.pagination.predicted_number,
                "pdf_label": page.pagination.pdf_label,
            },
            "elements": elements,
        })
    return {
        "source_path": document.source_path,
        "page_count": len(pages),
        "pages": pages,
    }


def _page_heading(page: dict[str, Any]) -> str:
    logical = page.get("nombre_text")
    if not logical and page.get("page_number") is not None:
        logical = str(page["page_number"])
    suffix = f" - {logical}" if logical else ""
    return f"Page {page['output_index'] + 1}{suffix}"


def export_markdown(document: Document) -> str:
    """Export OCR text and semantic non-text regions as portable Markdown."""
    data = document_to_data(document)
    lines = [f"# {Path(document.source_path).stem or 'Document'}", ""]
    for page in data["pages"]:
        lines.extend([f"## {_page_heading(page)}", ""])
        for element in page["elements"]:
            if element["type"] == RegionKind.TEXT.value:
                text = element.get("text", "").strip()
                if text:
                    lines.extend([text, ""])
                continue
            x0, y0, x1, y1 = element["bbox"]
            label = element.get("layout_label") or element["type"]
            lines.extend([
                f"> [{label}] bbox=({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})",
                "",
            ])
        lines.extend(["---", ""])
    return "\n".join(lines).rstrip() + "\n"


def export_html(document: Document) -> str:
    """Export semantic HTML without embedding or inventing image assets."""
    data = document_to_data(document)
    title = Path(document.source_path).stem or "Document"
    parts = [
        "<!doctype html>",
        '<html lang="ja">',
        "<head>",
        '  <meta charset="utf-8">',
        f"  <title>{html.escape(title)}</title>",
        "</head>",
        "<body>",
        f'<main data-source="{html.escape(document.source_path, quote=True)}">',
        f"<h1>{html.escape(title)}</h1>",
    ]
    for page in data["pages"]:
        parts.append(
            f'<section class="page" data-output-index="{page["output_index"]}" '
            f'data-source-page-index="{page["source_page_index"]}">'
        )
        parts.append(f"<h2>{html.escape(_page_heading(page))}</h2>")
        for element in page["elements"]:
            bbox = ",".join(str(value) for value in element["bbox"])
            label = element.get("layout_label") or element["type"]
            if element["type"] == RegionKind.TEXT.value:
                text = element.get("text", "").strip()
                if text:
                    parts.append(
                        f'<p data-order="{element["order"]}" data-bbox="{bbox}" '
                        f'data-role="{html.escape(label)}">{html.escape(text)}</p>'
                    )
            else:
                parts.append(
                    f'<figure data-order="{element["order"]}" data-bbox="{bbox}" '
                    f'data-kind="{html.escape(element["type"])}">'
                    f"<figcaption>{html.escape(label)}</figcaption></figure>"
                )
        parts.append("</section>")
    parts.extend(["</main>", "</body>", "</html>", ""])
    return "\n".join(parts)


def export_json(document: Document) -> str:
    return json.dumps(document_to_data(document), ensure_ascii=False, indent=2) + "\n"


def export_document(document: Document, output_format: str) -> str:
    normalized = output_format.lower().lstrip(".")
    if normalized in {"md", "markdown"}:
        return export_markdown(document)
    if normalized in {"htm", "html"}:
        return export_html(document)
    if normalized == "json":
        return export_json(document)
    raise ValueError(f"unsupported export format: {output_format}")


def infer_export_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    formats = {".md": "markdown", ".markdown": "markdown", ".html": "html",
               ".htm": "html", ".json": "json"}
    if suffix not in formats:
        raise ValueError("cannot infer format; use .md, .html, or .json output")
    return formats[suffix]


def write_export(document: Document, out_path: str, output_format: str | None = None) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_format or infer_export_format(out_path)
    path.write_text(export_document(document, fmt), encoding="utf-8")
    return str(path)
