"""Final rendering — the only stage that produces output pixels.

Principle (non-destructive, single resampling): deskew rotation, crop to the
normalized margin box, and rescale to the target dpi are composed into ONE
affine transform and applied to the ORIGINAL image in a single warp. Applying
them as separate steps would interpolate the image two or three times and
soften it; composing them means the original is sampled exactly once.

Coordinate convention: margin boxes and content regions are stored in the
deskewed-original frame (analysis already deskews the work raster). The same
composed transform maps those boxes analytically into output space, so no
second image pass is needed to know where text/figures land.

Image-layer encoding and the invisible searchable-text layer are delegated to
the validated hybrid-ocr pdf_export builder (CCITT G4 / JBIG2 / JPEG +
reading-order text). True multi-stream MRC (separate bilevel + photo layers
within one page) is the next step; v0 chooses the best per-page encoding,
which already gives small, searchable, clean output.
"""

from __future__ import annotations

import numpy as np
import cv2

from .model import Box, Document, PageKind, PageParams, RegionKind, RenderSettings


def compose_transform(
    original_shape: tuple[int, int],
    params: PageParams,
    settings: RenderSettings,
) -> tuple[np.ndarray, tuple[int, int], float]:
    """Return (M 2x3, output_size (w,h), scale).

    M maps original-image pixels directly to final output pixels.
    """
    h, w = original_shape[:2]
    angle = params.deskew.angle_deg
    # rotation about the original center (maps original -> deskewed frame)
    R = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    R_h = np.vstack([R, [0, 0, 1]])

    dpi = params.dpi or settings.target_dpi
    # crop box in deskewed frame. Prefer the normalized crop (uniform size,
    # nombre-anchored) when margin normalization has run; otherwise fall back
    # to the raw content box padded by the output margin.
    if params.margin and params.margin.crop:
        crop = params.margin.crop
        cx0, cy0, crop_w, crop_h = crop.x0, crop.y0, crop.width, crop.height
    elif params.margin and params.margin.content:
        c = params.margin.content
        margin_px = settings.output_margin_mm / 25.4 * dpi
        cx0, cy0 = c.x0 - margin_px, c.y0 - margin_px
        crop_w, crop_h = c.width + 2 * margin_px, c.height + 2 * margin_px
    else:
        cx0, cy0, crop_w, crop_h = 0, 0, w, h

    scale = settings.target_dpi / dpi if dpi else 1.0
    # scale + translate so (cx0,cy0) -> (0,0) and then upscale to target dpi
    A = np.array([[scale, 0, -scale * cx0],
                  [0, scale, -scale * cy0],
                  [0, 0, 1]], dtype=np.float64)
    M = (A @ R_h)[:2, :]
    out_size = (max(1, round(crop_w * scale)), max(1, round(crop_h * scale)))
    return M.astype(np.float32), out_size, scale


def _map_box(box: Box, M: np.ndarray) -> Box:
    pts = np.array([[box.x0, box.y0, 1], [box.x1, box.y1, 1]], dtype=np.float64).T
    out = M @ pts
    return Box(float(out[0, 0]), float(out[1, 0]), float(out[0, 1]), float(out[1, 1]))


def render_page_image(
    original_bgr: np.ndarray,
    params: PageParams,
    settings: RenderSettings,
) -> tuple[np.ndarray, list[dict], str, list[tuple]]:
    """Return (output_bgr, text_lines, page_mode, photo_boxes_px).

    output_bgr is the single-pass warped page. text_lines and photo_boxes are
    in OUTPUT pixels, ready for the searchable text overlay and MRC layering.
    Each photo box is a 5-tuple (x0, y0, x1, y1, tone) where tone is None
    (auto gray/color), "gray", or "color" — the per-region override that lets a
    grayscale picture live inside an otherwise bilevel page.
    """
    M, out_size, _ = compose_transform(original_bgr.shape, params, settings)
    out = cv2.warpAffine(original_bgr, M, out_size, flags=cv2.INTER_AREA,
                         borderValue=(255, 255, 255))

    lines = []
    photo_boxes: list[tuple] = []
    for r in params.regions:
        if r.kind == RegionKind.PHOTO:
            b = _map_box(r.box, M)
            photo_boxes.append((b.x0, b.y0, b.x1, b.y1, r.tone))
        if r.ocr_text:
            b = _map_box(r.box, M)
            lines.append({
                "box": [b.x0, b.y0, b.x1, b.y1],
                "polygon": [[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]],
                "text": r.ocr_text,
                "direction": "h" if b.width >= b.height else "v",
            })

    if params.page_kind in (PageKind.GRAY, PageKind.COLOR):
        mode = params.page_kind.value   # whole-page forced encoding
    else:
        # AUTO / BW / MRC -> bilevel base, photo regions become overlays
        mode = "bw"
    return out, lines, mode, photo_boxes


def build_pdf(
    document: Document,
    originals: list[np.ndarray],
    out_path: str,
) -> str:
    """Assemble the searchable MRC PDF from per-page params + originals."""
    from .mrc import MrcPageBuilder

    builder = MrcPageBuilder(
        compress=document.render.bilevel_codec,
        target_dpi=document.render.target_dpi,
        jpeg_quality=document.render.jpeg_quality,
    )
    for params, original in zip(document.pages, originals):
        out_bgr, lines, mode, photo_boxes = render_page_image(
            original, params, document.render
        )
        builder.add_page(out_bgr, lines, photo_boxes, mode)
    builder.save(out_path)
    _set_physical_page_size(out_path, document.render.target_dpi)
    return out_path


def _set_physical_page_size(pdf_path: str, dpi: int) -> None:
    """Rewrite each page's MediaBox from pixels (1px=1pt) to physical points
    at `dpi`, wrapping the content in a scale so the image still fills the
    page and the searchable-text layer stays aligned. Without this an ebook
    page reports a giant point size (e.g. 2820pt instead of ~340pt)."""
    import pikepdf

    s = 72.0 / dpi
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        for page in pdf.pages:
            mb = [float(v) for v in page.MediaBox]
            w_px, h_px = mb[2] - mb[0], mb[3] - mb[1]
            pg = pikepdf.Page(page)
            pg.contents_add(
                pikepdf.Stream(pdf, f"q {s} 0 0 {s} 0 0 cm".encode()),
                prepend=True,
            )
            pg.contents_add(pikepdf.Stream(pdf, b"Q"), prepend=False)
            page.MediaBox = [0, 0, round(w_px * s, 3), round(h_px * s, 3)]
        pdf.save()
