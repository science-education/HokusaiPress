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

# Tint-zone local thresholding constants.
# Text strokes on a tint background are pixels darker than the zone's tint
# median by at least this many grey levels.  Tuned for ADF-scanned halftone
# panels where the background peaks near 165-180 and text strokes are at 60-130.
_TINT_LO: int = 60
_TINT_HI: int = 220
_TINT_TEXT_DELTA: int = 40  # zone_median(tint pix) - 40 → local ink threshold


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
                 vector_fills: list | None = None,
                 tint_zones: list | None = None) -> None:
        from hybrid_ocr.pdf_export import encode_page_pdf

        from .render import binarize_bw

        h, w = out_bgr.shape[:2]
        vector_fills = vector_fills or []
        tint_zones = tint_zones or []
        tint_overlays: list[dict] = []

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
            # Tint zones: place a greyscale Multiply-blend overlay that preserves
            # the per-pixel tint colour.  Text strokes are identified with a
            # zone-LOCAL threshold (zone tint median - _TINT_TEXT_DELTA) rather
            # than the global Otsu threshold.
            #
            # Why not global Otsu?  The page-level Otsu is driven by the large
            # white-paper peak (mode ~240) and sets T ≈ 200-215, which binarises
            # the entire tint background as black -- exactly wrong for a
            # Multiply overlay.  The zone median of tint-range pixels
            # (60-220) gives a stable background estimate; pixels more than
            # _TINT_TEXT_DELTA below that are text strokes.
            for zone in tint_zones:
                x0, y0, x1, y1 = zone.rect
                x0i, y0i = max(0, int(x0)), max(0, int(y0))
                x1i, y1i = min(w, int(x1)), min(h, int(y1))
                if x1i - x0i < 4 or y1i - y0i < 4:
                    continue
                crop_gray = cv2.cvtColor(
                    out_bgr[y0i:y1i, x0i:x1i], cv2.COLOR_BGR2GRAY
                )
                tint_pix = crop_gray[
                    (crop_gray >= _TINT_LO) & (crop_gray <= _TINT_HI)
                ]
                if tint_pix.size >= 100:
                    zone_med = float(np.median(tint_pix))
                    ink_T = max(_TINT_LO, int(round(zone_med)) - _TINT_TEXT_DELTA)
                    ink = crop_gray < ink_T
                else:
                    ink = binary[y0i:y1i, x0i:x1i] == 0  # fallback
                # Build polygon mask: only modify pixels inside the contour shape.
                # Holes (e.g. the body-text cavity enclosed by a header/column/
                # footer frame) are punched out so they're left untouched.
                zone_h, zone_w = y1i - y0i, x1i - x0i
                if zone.page_contour is not None:
                    poly_mask = np.zeros((zone_h, zone_w), dtype=np.uint8)
                    cnt_crop = zone.page_contour.copy()
                    cnt_crop[:, 0, 0] -= x0i
                    cnt_crop[:, 0, 1] -= y0i
                    cv2.fillPoly(poly_mask, [cnt_crop], 255)
                    for hole in zone.hole_contours:
                        hole_crop = hole.copy()
                        hole_crop[:, 0, 0] -= x0i
                        hole_crop[:, 0, 1] -= y0i
                        cv2.fillPoly(poly_mask, [hole_crop], 0)
                    in_shape = poly_mask > 0
                else:
                    in_shape = np.ones((zone_h, zone_w), dtype=bool)
                ink_in_shape = ink & in_shape
                region = binary[y0i:y1i, x0i:x1i]
                region[in_shape & ~ink_in_shape] = 255   # tint bg → white
                region[ink_in_shape] = 0                  # text strokes → black
                # Overlay: greyscale, downsampled like photo overlays.
                ov_gray = zone.overlay_img              # HxW uint8
                ov_ds = cv2.resize(
                    ov_gray,
                    (max(1, int((x1i - x0i) * self.scale)),
                     max(1, int((y1i - y0i) * self.scale))),
                    interpolation=cv2.INTER_AREA,
                )
                tint_overlays.append({
                    "pdf": encode_page_pdf(
                        cv2.cvtColor(ov_ds, cv2.COLOR_GRAY2BGR), "gray", self.compress
                    ),
                    "rect": (x0i, h - y1i, x1i, h - y0i),  # PDF y-up
                    "contour": zone.page_contour,
                    "holes": zone.hole_contours,
                    "circle_fit": zone.circle_fit,
                    "page_h_px": h,
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
            "tint_overlays": tint_overlays,
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
                if page.get("tint_overlays"):
                    _add_tint_overlays(out, dest, page["tint_overlays"], i, sources)
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


def _add_tint_overlays(
    pdf, page, tint_ovs: list, page_idx: int, sources: list
) -> None:
    """Place greyscale tint overlays with /BM /Multiply blend mode.

    Each overlay is a greyscale JPEG image stored in a single-page PDF.
    Multiply blend: overlay_value × bilevel_white = overlay_value (tint shows),
    overlay_value × bilevel_black = 0 (text strokes stay black regardless).
    Paper pixels are 255 in the overlay, so Multiply(1.0, bilevel) = bilevel
    (no change on paper).

    When a contour is present the overlay is clipped to the exact tint-shape
    polygon via a PDF clip path (W/W* n operator) before the XObject is
    painted.  This eliminates the black fringe that occurs when the bounding
    rect extends beyond the actual tint region (e.g. a circular badge or a
    rounded corner), and -- when the zone has holes (a header/column/footer
    frame enclosing the body text) -- keeps the enclosed cavity unpainted.
    """
    import pikepdf

    from .tint_zone import circle_to_pdf_path, contour_to_pdf_path

    resources = page.Resources
    if "/ExtGState" not in resources:
        resources.ExtGState = pikepdf.Dictionary()
    resources.ExtGState[pikepdf.Name("/HPVecFillMultiply")] = pikepdf.Dictionary({
        "/Type": pikepdf.Name("/ExtGState"),
        "/BM": pikepdf.Name("/Multiply"),
    })

    p = pikepdf.Page(page)
    for i, ov in enumerate(tint_ovs):
        ovpdf = pikepdf.open(BytesIO(ov["pdf"]))
        sources.append(ovpdf)
        form = pikepdf.Page(ovpdf.pages[0]).as_form_xobject()
        name = pikepdf.Name(f"/HPTintOv{page_idx}_{i}")
        placed = p.add_resource(form, pikepdf.Name.XObject, name=name)
        rect = pikepdf.Rectangle(*ov["rect"])
        inner = p.calc_form_xobject_placement(
            form, placed, rect, allow_shrink=True, allow_expand=True
        )
        # Nested q/Q: outer activates Multiply; clip path (if any) restricts
        # painting to the actual tint polygon so corners/badges don't overflow.
        # Circles use 4-arc Bézier for mathematical precision; polygons use
        # approxPolyDP-simplified m/l/h paths.
        # Holes (e.g. the body-text cavity enclosed by a header/column/footer
        # frame) are appended as extra subpaths in the same clip path, and
        # the clip operator switches to "W*" (even-odd) so they're excluded
        # from the painted region -- the standard PDF "donut clip" technique.
        if ov.get("circle_fit") is not None:
            cx, cy, r = ov["circle_fit"]
            clip = circle_to_pdf_path(cx, cy, r, ov["page_h_px"])
            clip_op = b"W n\n"
        elif ov.get("contour") is not None:
            holes = ov.get("holes") or []
            clip = contour_to_pdf_path(ov["contour"], ov["page_h_px"], holes)
            clip_op = b"W* n\n" if holes else b"W n\n"
        else:
            clip = b""
            clip_op = b"W n\n"
        if clip:
            content = b"q /HPVecFillMultiply gs\n" + clip + clip_op + inner + b"\nQ\n"
        else:
            content = b"q /HPVecFillMultiply gs\n" + inner + b"\nQ\n"
        p.contents_add(pikepdf.Stream(pdf, content), prepend=False)
        p.contents_coalesce()


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
