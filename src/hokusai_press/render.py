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
    # All FOUR corners, not just the two diagonal ones: M can carry a deskew
    # rotation, under which a rotated rectangle's true bounding box is not
    # simply "transform the top-left and bottom-right corners" -- a corner
    # far from the rotation center (e.g. a text line near the very top of a
    # tall page) shifts sideways by several px more than the box's nominal
    # opposite corner does. Mapping only 2 corners under-covered exactly that
    # far corner, and the margin-fill step then whitened part of real text
    # past the (wrongly short) computed edge -- seen on real corpus data
    # (img20260427_0001 p11: deskew -0.4 deg clipped the right side of
    # characters in a text line near the page top, far from the rotation
    # center, while center-ish lines were unaffected).
    pts = np.array([
        [box.x0, box.y0, 1], [box.x1, box.y0, 1],
        [box.x0, box.y1, 1], [box.x1, box.y1, 1],
    ], dtype=np.float64).T
    out = M @ pts
    return Box(float(out[0].min()), float(out[1].min()),
               float(out[0].max()), float(out[1].max()))


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
    from .geometry.margin import remove_edge_shadows, remove_region_shadows

    # whiten binding/ADF edge shadows on the FULL original first (robust, the
    # validated location) -- then warp the clean image, so no shadow survives
    # into the output and a wrong crop can't reintroduce one.
    M, out_size, _ = compose_transform(original_bgr.shape, params, settings)
    # OCR-confident blank page: emit pure white (no warp needed), so neither
    # show-through nor any residual speck reaches the output.
    if getattr(params, "blank", False):
        white = np.full((out_size[1], out_size[0], 3), 255, dtype=np.uint8)
        return white, [], "bw", []

    # region-based margin shadow removal first (uses this page's OCR text/figure
    # regions: thin near-full-height dark line outside the text box, figures
    # protected), then the per-page component rule for wider shadows. On the
    # production run path the original is already region-cleaned (no-op here); on
    # the rebuild path the original is freshly loaded, so this is where it applies.
    clean = remove_region_shadows(original_bgr, params.regions)
    clean = remove_edge_shadows(clean)
    out = cv2.warpAffine(clean, M, out_size, flags=cv2.INTER_AREA,
                         borderValue=(255, 255, 255))

    # Fill the margin (everything outside the content box) with white. The
    # content box already excludes edge lines/shadows, so this deterministically
    # removes any hard border the uniform-size crop pulled in (ADF scans: white
    # paper, a thin dark edge line outside the content). Text/figures/photos are
    # inside the content box, so nothing real is touched. ScanTailor's "fill
    # margins", and the right tool for uniform-illumination ADF scans.
    if params.margin and params.margin.content:
        ow, oh = out_size
        b = _map_box(params.margin.content, M)
        x0 = max(0, min(ow, int(round(b.x0))))
        y0 = max(0, min(oh, int(round(b.y0))))
        x1 = max(0, min(ow, int(round(b.x1))))
        y1 = max(0, min(oh, int(round(b.y1))))
        out[:y0, :] = 255
        out[y1:, :] = 255
        out[:, :x0] = 255
        out[:, x1:] = 255

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


def _figure_vecfills(
    out_bgr: np.ndarray,
    params: PageParams,
    settings: RenderSettings,
    original_shape: tuple[int, int],
) -> list[dict]:
    """Return vector fills for solid FIGURE regions in rendered pixel space."""
    from .region_class import classify_patch

    M, _, _ = compose_transform(original_shape, params, settings)
    h, w = out_bgr.shape[:2]
    fills = []
    for r in params.regions:
        if r.kind != RegionKind.FIGURE:
            continue
        b = _map_box(r.box, M)
        x0i = max(0, min(w, int(round(min(b.x0, b.x1)))))
        y0i = max(0, min(h, int(round(min(b.y0, b.y1)))))
        x1i = max(0, min(w, int(round(max(b.x0, b.x1)))))
        y1i = max(0, min(h, int(round(max(b.y0, b.y1)))))
        if x1i - x0i < 4 or y1i - y0i < 4:
            continue
        patch = out_bgr[y0i:y1i, x0i:x1i]
        if classify_patch(patch) != "solid_fill":
            continue
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Median is stable under small antialiasing or scan noise at the edge.
        tone = float(np.median(gray)) / 255.0
        fills.append({"rect": (x0i, y0i, x1i, y1i), "gray": tone})
    return fills


def _overlaps_any(rect: tuple, fills: list[dict]) -> bool:
    """True if rect shares >50% of its area with any existing fill."""
    x0, y0, x1, y1 = rect
    area = max(1, (x1 - x0) * (y1 - y0))
    for f in fills:
        fx0, fy0, fx1, fy1 = f["rect"]
        ix = max(0, min(x1, fx1) - max(x0, fx0))
        iy = max(0, min(y1, fy1) - max(y0, fy0))
        if ix * iy > area * 0.5:
            return True
    return False


def _raster_tint_fills(out_bgr: np.ndarray, existing: list[dict]) -> list[dict]:
    """Detect tint panels from the raster and return fills not already covered.

    Legacy flat-fill path kept for tests.  Production code uses _raster_tint_zones.
    """
    from .tint_panel import detect_tint_panels

    new_fills = []
    for fill in detect_tint_panels(out_bgr):
        if not _overlaps_any(fill["rect"], existing):
            new_fills.append(fill)
    return new_fills


def _raster_tint_zones(
    out_bgr: np.ndarray,
    existing_fills: list[dict],
    text_boxes: list[tuple[float, float, float, float]] | None = None,
):
    """Detect tint panels and return non-overlapping TintZone objects.

    Panels that overlap with already-placed vector fills are skipped.
    Column-projection results that overlap with row-projection results at their
    ends are clipped inside build_tint_zones (overlap deduplication).
    """
    from .tint_panel import detect_tint_panels
    from .tint_zone import build_tint_zones

    import cv2

    gray = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY) if out_bgr.ndim == 3 else out_bgr
    panels = [
        p for p in detect_tint_panels(out_bgr)
        if not _overlaps_any(p["rect"], existing_fills)
    ]
    return build_tint_zones(gray, panels, text_boxes)


def binarize_bw(bgr: np.ndarray, book_valley: "int | None" = None) -> np.ndarray:
    """{0,255} single-channel bw layer. Threshold is Otsu (the real ink/paper
    valley, kept as is so light strokes survive); on a degenerate near-blank page
    -- high Otsu with no genuinely dark pixels -- ink_threshold drops to the
    book's valley (2-pass, when book_valley is given) or a fixed floor so
    show-through / shadow penumbra stay white instead of turning black (see
    geometry.margin.ink_threshold). HokusaiPress owns its output binarization
    (the OCR side keeps its own global-Otsu binarize for recognition)."""
    from .geometry.margin import ink_threshold

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return np.where(gray <= ink_threshold(gray, book_valley), 0, 255).astype(np.uint8)


def _binarize_for_preview(bgr: np.ndarray) -> np.ndarray:
    return binarize_bw(bgr)


def render_output_preview(
    original_bgr: np.ndarray,
    params: PageParams,
    settings: RenderSettings,
) -> np.ndarray:
    """A BGR image that mirrors what the final PDF page will actually contain:
    deskewed + cropped + dpi-normalized, then encoded the way the MRC builder
    would — bilevel base with photo regions kept as gray/color overlays, or a
    whole-page gray/color image when page_kind forces it. This is what the
    review UI labels "output" (vs. the raw warped image, which hid binarization).
    """
    out, _, mode, photo_boxes = render_page_image(original_bgr, params, settings)
    if mode == "gray":
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if mode == "color":
        return out

    # bw / MRC: binarize the base, then paste photo regions back as gray/color
    from .mrc import _is_grayish

    h, w = out.shape[:2]
    canvas = cv2.cvtColor(_binarize_for_preview(out), cv2.COLOR_GRAY2BGR)
    for (x0, y0, x1, y1, tone) in photo_boxes:
        x0i, y0i = max(0, int(x0)), max(0, int(y0))
        x1i, y1i = min(w, int(x1)), min(h, int(y1))
        if x1i - x0i < 4 or y1i - y0i < 4:
            continue
        crop = out[y0i:y1i, x0i:x1i]
        cmode = tone or ("gray" if _is_grayish(crop) else "color")
        if cmode == "gray":
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        canvas[y0i:y1i, x0i:x1i] = crop
    return canvas


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
        ink_valley=getattr(document.render, "ink_valley", None),
    )
    for params, original in zip(document.pages, originals):
        out_bgr, lines, mode, photo_boxes = render_page_image(
            original, params, document.render
        )
        vecfills = _figure_vecfills(out_bgr, params, document.render, original.shape)
        text_boxes = [tuple(line["box"]) for line in lines]
        tint_zones = (
            _raster_tint_zones(out_bgr, vecfills, text_boxes)
            if document.render.tint_overlay else []
        )
        builder.add_page(out_bgr, lines, photo_boxes, mode, vecfills, tint_zones)
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
        pdf.save(deterministic_id=True)
