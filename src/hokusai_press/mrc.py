"""Mixed Raster Content (MRC) page assembly.

A page with photos/figures should not be stored as one big JPEG (text turns
soft, file gets heavy) nor as one bilevel image (halftones turn to mud).
MRC splits each page into layers in a single PDF page:

- base layer: the whole page binarized to crisp G4/JBIG2, with photo regions
  erased to white so halftone never gets binarized;
- photo layers: each photo region cropped from the full-resolution render and
  down-sampled to a modest dpi, encoded as JPEG and overlaid at its location;
- text layer: the invisible searchable layer.

This keeps characters razor-sharp and light while photos stay legible at a
fraction of the bytes. All validated encoders are reused from hybrid-ocr's
pdf_export; this module only composes their single-page PDFs with pikepdf.

Compositing is done in pixel space (1px = 1pt); the caller rescales MediaBoxes
to physical points afterwards.
"""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np


class MrcPageBuilder:
    """Accumulates pages and writes one searchable MRC PDF."""

    def __init__(self, compress: str = "g4", photo_dpi: int = 200,
                 target_dpi: int = 600, jpeg_quality: int = 85):
        self.compress = compress
        self.scale = max(photo_dpi / target_dpi, 0.05)
        self.jpeg_quality = jpeg_quality
        self._pages: list[dict] = []

    def add_page(self, out_bgr: np.ndarray, lines: list,
                 photo_boxes_px: list, mode: str) -> None:
        from hybrid_ocr.pdf_export import encode_page_pdf

        from .render import binarize_bw

        h, w = out_bgr.shape[:2]
        if mode in ("gray", "color"):
            base_pdf = encode_page_pdf(out_bgr, mode, self.compress)
            overlays = []
        else:
            binary = binarize_bw(out_bgr)         # {0,255}, clamped threshold
            overlays = []
            for box in photo_boxes_px:
                x0, y0, x1, y1 = box[:4]
                tone = box[4] if len(box) > 4 else None  # None|"gray"|"color"
                x0i, y0i = max(0, int(x0)), max(0, int(y0))
                x1i, y1i = min(w, int(x1)), min(h, int(y1))
                if x1i - x0i < 4 or y1i - y0i < 4:
                    continue
                binary[y0i:y1i, x0i:x1i] = 255   # erase photo from bilevel
                crop = out_bgr[y0i:y1i, x0i:x1i]
                ds = cv2.resize(
                    crop,
                    (max(1, int((x1i - x0i) * self.scale)),
                     max(1, int((y1i - y0i) * self.scale))),
                    interpolation=cv2.INTER_AREA,
                )
                # explicit per-region override wins; else decide from chroma
                cmode = tone or ("gray" if _is_grayish(crop) else "color")
                overlays.append({
                    "pdf": encode_page_pdf(ds, cmode, self.compress),
                    "rect": (x0i, h - y1i, x1i, h - y0i),  # PDF y-up
                })
            base_pdf = encode_page_pdf(binary, "bw", self.compress)

        self._pages.append({"base": base_pdf, "overlays": overlays,
                            "w": w, "h": h, "lines": lines})

    def save(self, output_path: str) -> None:
        import pikepdf
        from hybrid_ocr.pdf_export import build_text_overlay

        if not self._pages:
            raise ValueError("no pages added")

        text_pdf_bytes = build_text_overlay(
            [(p["w"], p["h"], p["lines"]) for p in self._pages]
        )
        out = pikepdf.new()
        sources = []
        try:
            text_pdf = pikepdf.open(BytesIO(text_pdf_bytes))
            sources.append(text_pdf)
            for i, page in enumerate(self._pages):
                base = pikepdf.open(BytesIO(page["base"]))
                sources.append(base)
                out.pages.extend(base.pages)
                dest = out.pages[-1]
                for ov in page["overlays"]:
                    ovpdf = pikepdf.open(BytesIO(ov["pdf"]))
                    sources.append(ovpdf)
                    pikepdf.Page(dest).add_overlay(
                        ovpdf.pages[0], pikepdf.Rectangle(*ov["rect"])
                    )
                pikepdf.Page(dest).add_overlay(text_pdf.pages[i])
            out.save(output_path)
        finally:
            for s in sources:
                s.close()


def _is_grayish(bgr: np.ndarray) -> bool:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    chroma = lab[..., 1].astype(np.float32).std() + lab[..., 2].astype(np.float32).std()
    return chroma <= 16.0
