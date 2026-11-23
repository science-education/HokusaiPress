"""`hokusai-press ocr`: replace the text layer of finished PDFs.

For PDFs that are already corrected/binarized (by an earlier run, ScanTailor,
...): every page is rendered at `dpi`, OCR'd with the selected engine, and a
new invisible text layer is written over the UNCHANGED source pages, so the
G4/JBIG2/JPEG images are not re-encoded. The old text layer is removed first
(unless keep_existing_text).

One OCR engine instance (get_ocr_engine's process-wide cache) serves all
files, so models load once per command, not once per PDF.
"""

from __future__ import annotations

import os
import time

import numpy as np

RENDER_DPI = 300


def _render_pages(path: str, dpi: int):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=dpi / 72.0)
            rgb = np.asarray(bitmap.to_pil().convert("RGB"))
            bitmap.close()
            page.close()
            yield np.ascontiguousarray(rgb[:, :, ::-1])  # BGR for the engines
    finally:
        pdf.close()


def ocr_pdf(src: str, dst: str, engine, dpi: int = RENDER_DPI,
            keep_existing_text: bool = False, text_out: str | None = None,
            progress=None) -> dict:
    """OCR src page by page and write dst with a fresh text layer.

    Returns {"pages", "lines", "chars", "seconds"}."""
    from .model import ensure_parent_dir, refuse_overwrite
    from .textlayer import check_no_rotated_pages, overlay_text

    refuse_overwrite(src, dst)
    if text_out:
        refuse_overwrite(src, text_out)
        refuse_overwrite(dst, text_out)
    check_no_rotated_pages(src)
    ensure_parent_dir(dst)
    if text_out:
        ensure_parent_dir(text_out)

    t0 = time.perf_counter()
    pages, texts = [], []
    n_lines = n_chars = 0
    for i, img in enumerate(_render_pages(src, dpi)):
        result = engine(img)
        lines = []
        for ln in result.get("lines", []):
            text = ln.get("text") or ""
            x0, y0, x1, y1 = ln["box"]
            lines.append({"box": (x0, y0, x1, y1), "text": text})
            n_chars += len(text)
        n_lines += len(lines)
        h, w = img.shape[:2]
        pages.append((w, h, lines))
        texts.append("\n".join(ln["text"] for ln in lines))
        if progress:
            progress(i + 1)
    overlay_text(src, pages, dst, strip_existing=not keep_existing_text)
    if text_out:
        with open(text_out, "w", encoding="utf-8") as f:
            f.write("\f".join(texts))
    return {"pages": len(pages), "lines": n_lines, "chars": n_chars,
            "seconds": time.perf_counter() - t0}


def output_path(out: str, src: str, suffix: str, is_folder: bool) -> str:
    if not is_folder:
        return out
    os.makedirs(out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(out, stem + suffix + ".pdf")
