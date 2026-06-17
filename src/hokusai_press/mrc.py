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
                 photo_boxes_px: list, mode: str,
                 vector_fills: list | None = None) -> None:
        from hybrid_ocr.pdf_export import encode_page_pdf

        from .render import binarize_bw

        h, w = out_bgr.shape[:2]
        vector_fills = vector_fills or []
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
            for fill in vector_fills:
                x0, y0, x1, y1 = fill["rect"]
                x0i, y0i = max(0, int(x0)), max(0, int(y0))
                x1i, y1i = min(w, int(x1)), min(h, int(y1))
                if x1i - x0i < 1 or y1i - y0i < 1:
                    continue
                crop = out_bgr[y0i:y1i, x0i:x1i]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                ink = gray < max(0, int(round(fill["gray"] * 255)) - 50)
                binary[y0i:y1i, x0i:x1i] = 255
                region = binary[y0i:y1i, x0i:x1i]
                region[ink] = 0
            base_pdf = encode_page_pdf(binary, "bw", self.compress)

        self._pages.append({
            "base": base_pdf,
            "overlays": overlays,
            "vector_fills": vector_fills,
            "w": w,
            "h": h,
            "lines": lines,
        })

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
                for j, ov in enumerate(page["overlays"]):
                    ovpdf = pikepdf.open(BytesIO(ov["pdf"]))
                    sources.append(ovpdf)
                    _add_overlay_named(
                        dest,
                        ovpdf.pages[0],
                        pikepdf.Rectangle(*ov["rect"]),
                        f"/HPPhoto{i}_{j}",
                    )
                if page["vector_fills"]:
                    _add_vector_fills(out, dest, page["vector_fills"], page["h"])
                _add_overlay_named(dest, text_pdf.pages[i], None, f"/HPText{i}")
            out.save(output_path, deterministic_id=True)
        finally:
            for s in sources:
                s.close()


def _is_grayish(bgr: np.ndarray) -> bool:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    chroma = lab[..., 1].astype(np.float32).std() + lab[..., 2].astype(np.float32).std()
    return chroma <= 16.0


def _add_overlay_named(dest, overlay_page, rect, name: str) -> None:
    import pikepdf

    page = pikepdf.Page(dest)
    form = pikepdf.Page(overlay_page).as_form_xobject()
    placed = page.add_resource(form, pikepdf.Name.XObject, name=pikepdf.Name(name))
    if rect is None:
        rect = pikepdf.Rectangle(page.trimbox)
    content = page.calc_form_xobject_placement(
        form, placed, rect, allow_shrink=True, allow_expand=True
    )
    page.contents_add(b"q\n", prepend=True)
    page.contents_add(b"Q\n", prepend=False)
    page.contents_add(content, prepend=False)
    page.contents_coalesce()


def _add_vector_fills(pdf, page, vector_fills: list, page_h_px: int) -> None:
    import pikepdf

    from .vecfill import fill_rect_ops, px_to_pt

    resources = page.Resources
    if "/ExtGState" not in resources:
        resources.ExtGState = pikepdf.Dictionary()
    resources.ExtGState[pikepdf.Name("/HPVecFillMultiply")] = pikepdf.Dictionary({
        "/Type": pikepdf.Name("/ExtGState"),
        "/BM": pikepdf.Name("/Multiply"),
    })

    scale = px_to_pt(1, 72)
    parts = [b"q /HPVecFillMultiply gs\n"]
    for fill in vector_fills:
        x0, y0, x1, y1 = fill["rect"]
        parts.append(fill_rect_ops(x0, y0, x1, y1, fill["gray"], page_h_px, scale))
    parts.append(b"Q\n")
    pikepdf.Page(page).contents_add(pikepdf.Stream(pdf, b"".join(parts)))
