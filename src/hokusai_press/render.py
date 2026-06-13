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

    # crop box in deskewed frame: content padded by the normalized margin
    if params.margin and params.margin.content:
        c = params.margin.content
    else:
        c = Box(0, 0, w, h)
    dpi = params.dpi or settings.target_dpi
    margin_px = settings.output_margin_mm / 25.4 * dpi
    cx0 = c.x0 - margin_px
    cy0 = c.y0 - margin_px
    crop_w = c.width + 2 * margin_px
    crop_h = c.height + 2 * margin_px

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
) -> tuple[np.ndarray, list[dict], str]:
    """Return (output_bgr, text_lines_for_pdf, page_mode).

    output_bgr is the single-pass warped page. text_lines are in OUTPUT
    pixels, ready for the searchable text overlay.
    """
    M, out_size, _ = compose_transform(original_bgr.shape, params, settings)
    out = cv2.warpAffine(original_bgr, M, out_size, flags=cv2.INTER_AREA,
                         borderValue=(255, 255, 255))

    lines = []
    has_photo = False
    for r in params.regions:
        if r.kind == RegionKind.PHOTO:
            has_photo = True
        if r.ocr_text:
            b = _map_box(r.box, M)
            lines.append({
                "box": [b.x0, b.y0, b.x1, b.y1],
                "polygon": [[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]],
                "text": r.ocr_text,
                "direction": "h" if b.width >= b.height else "v",
            })

    if params.page_kind != PageKind.AUTO:
        mode = params.page_kind.value if params.page_kind != PageKind.MRC else "bw"
    else:
        mode = "auto" if has_photo else "bw"
    return out, lines, mode


def build_pdf(
    document: Document,
    originals: list[np.ndarray],
    out_path: str,
) -> str:
    """Assemble the searchable PDF from per-page params + originals."""
    from hybrid_ocr.pdf_export import SearchablePdfBuilder

    builder = SearchablePdfBuilder(
        mode="auto", compress=document.render.bilevel_codec
    )
    for params, original in zip(document.pages, originals):
        out_bgr, lines, mode = render_page_image(original, params, document.render)
        builder.add_page(out_bgr, lines, mode=None if mode == "auto" else mode)
    builder.save(out_path)
    return out_path
