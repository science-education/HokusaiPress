"""Unit tests for automatic tone restoration (synthetic pages)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pikepdf")
pdfium = pytest.importorskip("pypdfium2")

from hokusai_press.model import (
    Box,
    Deskew,
    Document,
    Margin,
    PageKind,
    PageParams,
    Region,
    RegionKind,
    RenderSettings,
    SourceRef,
)
from hokusai_press.render import (
    _prepare_bilevel_layers,
    build_pdf,
    render_output_preview,
    render_page_image,
)


def _make_params(
    w: int,
    h: int,
    regions: list[Region] | None = None,
    page_kind: PageKind = PageKind.AUTO,
    tinted: bool = False,
) -> PageParams:
    return PageParams(
        source=SourceRef(path="synthetic.png", page_index=0),
        dpi=144.0,
        deskew=Deskew(angle_deg=0.0, confidence=5.0),
        margin=Margin(
            content=Box(0, 0, w, h),
            confidence=1.0,
            page_w=float(w),
            page_h=float(h),
            tinted=tinted,
        ),
        regions=regions or [],
        page_kind=page_kind,
    )


def _add_horizontal_body_lines(
    img: np.ndarray,
    y_start: int,
    y_end: int,
    x0: int = 80,
    x1: int = 620,
    step: int = 28,
) -> list[Region]:
    regions: list[Region] = []
    for y in range(y_start, y_end, step):
        for x in range(x0, x1 - 10, 14):
            img[y:y + 8, x:x + 9] = 25
        regions.append(
            Region(
                kind=RegionKind.TEXT,
                box=Box(x0, y, x1, y + 10),
                ocr_text="これは通常の本文行であり十分な文字数があります。",
            )
        )
    return regions


def _add_vertical_body_lines(
    img: np.ndarray,
    x_start: int,
    x_end: int,
    y0: int = 100,
    y1: int = 820,
    step: int = 32,
) -> list[Region]:
    regions: list[Region] = []
    for x in range(x_start, x_end, step):
        for y in range(y0, y1 - 10, 14):
            img[y:y + 9, x:x + 8] = 25
        regions.append(
            Region(
                kind=RegionKind.TEXT,
                box=Box(x, y0, x + 10, y1),
                ocr_text="これは縦書きの本文列であり十分な文字数があります。",
            )
        )
    return regions


def test_gray_banner_with_knockout_text_is_restored():
    """A dark-ish gray banner with white knockout text that crushes to black in Otsu is restored."""
    w, h = 700, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Dark gray header banner (tone ~95) + white knockout text
    img[180:250, 100:600] = 95
    for x in range(140, 560, 26):
        img[198:232, x:x + 14] = 252
    # Add dark ink elsewhere so Otsu threshold sits near ~140 (crushing 95 -> 0)
    regions = _add_horizontal_body_lines(img, 360, 860)
    params = _make_params(w, h, regions=regions)
    settings = RenderSettings(target_dpi=144, output_margin_mm=0.0)

    out_bgr, lines, _, photo_boxes = render_page_image(img, params, settings)
    _, _, tone_alpha, canvas, restore = _prepare_bilevel_layers(
        out_bgr, params, settings, lines, photo_boxes, img.shape
    )
    assert restore.adopted is True
    assert tone_alpha is not None
    # Banner interior should remain near 95 rather than crushed to 0
    banner_med = float(np.median(canvas[190:240, 120:135, 0]))
    assert 70 <= banner_med <= 125
    # Body text area outside mask remains strictly bilevel {0, 255}
    body_vals = np.unique(canvas[400:800, 80:620])
    assert set(body_vals.tolist()).issubset({0, 255})


def test_diagram_mixed_with_body_text_preserves_body_pixels():
    """Restoring a diagram banner mixed with body text leaves outside body pixels untouched."""
    w, h = 700, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    regions = _add_horizontal_body_lines(img, 100, 320)
    # Pale gray diagram box (tone ~210, whites out in Otsu) in middle
    img[400:540, 120:580] = 210
    img[430:510, 160:540] = 175
    regions.extend(_add_horizontal_body_lines(img, 620, 880))

    params = _make_params(w, h, regions=regions)
    settings = RenderSettings(target_dpi=144, output_margin_mm=0.0)
    out_bgr, lines, _, photo_boxes = render_page_image(img, params, settings)
    _, _, tone_alpha, canvas, restore = _prepare_bilevel_layers(
        out_bgr, params, settings, lines, photo_boxes, img.shape
    )
    assert restore.adopted is True
    assert tone_alpha is not None
    assert float(np.median(canvas[440:500, 200:500, 0])) < 200
    # Top and bottom body text regions remain pure bilevel
    assert set(np.unique(canvas[100:320, :]).tolist()).issubset({0, 255})
    assert set(np.unique(canvas[620:880, :]).tolist()).issubset({0, 255})


def test_partial_diagram_rects_are_grouped_into_single_region():
    """Adjacent parts of the same diagram (two boxes with a disconnected arrow between) merge."""
    w, h = 700, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Two gray boxes in the same diagram separated vertically, with a thin disconnected arrow in between
    img[200:255, 120:540] = 205
    img[310:365, 140:560] = 205
    img[270:295, 338:342] = 180  # thin disconnected arrow shaft in the gap
    regions = _add_horizontal_body_lines(img, 500, 860)

    params = _make_params(w, h, regions=regions)
    settings = RenderSettings(target_dpi=144, output_margin_mm=0.0)
    out_bgr, lines, _, photo_boxes = render_page_image(img, params, settings)
    _, _, tone_alpha, _, restore = _prepare_bilevel_layers(
        out_bgr, params, settings, lines, photo_boxes, img.shape
    )
    assert restore.adopted is True
    assert len(restore.expanded_regions) >= 1
    assert tone_alpha is not None
    # The gap containing the faint arrow is covered by the grouped mask
    assert float(tone_alpha[282, 340]) > 0.5


def test_separate_diagrams_stay_independent():
    """Two diagrams separated by a large vertical gap are not merged across the gap."""
    w, h = 700, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Top diagram
    img[120:220, 120:560] = 200
    # Body text between them
    regions = _add_horizontal_body_lines(img, 320, 560)
    # Bottom diagram
    img[660:760, 120:560] = 200

    params = _make_params(w, h, regions=regions)
    settings = RenderSettings(target_dpi=144, output_margin_mm=0.0)
    out_bgr, lines, _, photo_boxes = render_page_image(img, params, settings)
    _, _, tone_alpha, _, restore = _prepare_bilevel_layers(
        out_bgr, params, settings, lines, photo_boxes, img.shape
    )
    assert restore.adopted is True
    assert tone_alpha is not None
    # Body text gap between the two diagrams must stay outside the mask
    assert float(tone_alpha[400, 350]) == 0.0


def test_dense_horizontal_body_page_is_not_modified():
    """Normal dense horizontal body text must not trigger tone restoration."""
    w, h = 700, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    regions = _add_horizontal_body_lines(img, 80, 900, step=22)

    params = _make_params(w, h, regions=regions)
    settings = RenderSettings(target_dpi=144, output_margin_mm=0.0)
    out_bgr, lines, _, photo_boxes = render_page_image(img, params, settings)
    _, _, tone_alpha, _, restore = _prepare_bilevel_layers(
        out_bgr, params, settings, lines, photo_boxes, img.shape
    )
    assert restore.adopted is False
    assert tone_alpha is None


def test_vertical_body_text_page_is_not_modified():
    """Vertical body text columns must not trigger tone restoration."""
    w, h = 700, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    regions = _add_vertical_body_lines(img, 90, 610, step=26)

    params = _make_params(w, h, regions=regions)
    settings = RenderSettings(target_dpi=144, output_margin_mm=0.0)
    out_bgr, lines, _, photo_boxes = render_page_image(img, params, settings)
    _, _, tone_alpha, _, restore = _prepare_bilevel_layers(
        out_bgr, params, settings, lines, photo_boxes, img.shape
    )
    assert restore.adopted is False
    assert tone_alpha is None


def test_vecfill_is_not_double_drawn_on_restored_gray_region():
    """A FIGURE vecfill overlapping the restored region must be suppressed so tone is not darkened."""
    w, h = 700, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Gray box (tone 160) with a darker inner band (tone 90) so Otsu crushes/splits
    img[200:320, 120:580] = 160
    img[200:230, 120:580] = 90
    regions = _add_horizontal_body_lines(img, 450, 860)
    # Mark the lower solid-gray part as a FIGURE region that would trigger vecfill
    regions.append(Region(kind=RegionKind.FIGURE, box=Box(120, 235, 580, 320)))

    params = _make_params(w, h, regions=regions)
    settings = RenderSettings(target_dpi=144, output_margin_mm=0.0)
    out_bgr, lines, _, photo_boxes = render_page_image(img, params, settings)
    vecfills, _, tone_alpha, canvas, restore = _prepare_bilevel_layers(
        out_bgr, params, settings, lines, photo_boxes, img.shape
    )
    assert restore.adopted is True
    assert tone_alpha is not None
    # Overlapping vecfill must have been removed
    assert vecfills == []
    # Gray tone stays near 160, NOT darkened to ~160 * 0.63 = 100
    restored_med = float(np.median(canvas[250:300, 200:500, 0]))
    assert 145 <= restored_med <= 175


def test_preview_and_pdf_paths_match_and_respect_manual_modes(tmp_path):
    """Output preview and built PDF agree on tone restoration, and manual gray/tinted are untouched."""
    w, h = 700, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[180:250, 100:600] = 95
    for x in range(140, 560, 26):
        img[198:232, x:x + 14] = 252
    regions = _add_horizontal_body_lines(img, 360, 860)

    params = _make_params(w, h, regions=regions)
    settings = RenderSettings(target_dpi=144, output_margin_mm=0.0)
    preview = render_output_preview(img, params, settings)

    out_pdf = tmp_path / "out.pdf"
    doc = Document(source_path="synthetic.png", pages=[params], render=settings)
    build_pdf(doc, [img], str(out_pdf))

    pdf_doc = pdfium.PdfDocument(str(out_pdf))
    try:
        page = pdf_doc[0]
        bitmap = page.render(scale=144.0 / 72.0)
        pdf_img = bitmap.to_numpy()
        # Resize only if rounding differs by 1px
        if pdf_img.shape[:2] != preview.shape[:2]:
            import cv2
            pdf_img = cv2.resize(pdf_img, (preview.shape[1], preview.shape[0]))
        banner_prev = float(np.median(preview[190:240, 120:135, 0]))
        banner_pdf = float(np.median(pdf_img[190:240, 120:135, 0]))
        assert abs(banner_prev - banner_pdf) <= 10.0
    finally:
        pdf_doc.close()

    # Manual GRAY / COLOR / tinted pages are not overridden
    gray_params = _make_params(w, h, regions=regions, page_kind=PageKind.GRAY)
    out_g, lines_g, mode_g, _ = render_page_image(img, gray_params, settings)
    assert mode_g == "gray"
    tinted_params = _make_params(w, h, regions=regions, tinted=True)
    out_t, lines_t, _, photo_t = render_page_image(img, tinted_params, settings)
    _, _, alpha_t, _, restore_t = _prepare_bilevel_layers(
        out_t, tinted_params, settings, lines_t, photo_t, img.shape
    )
    assert restore_t.adopted is False
    assert alpha_t is None


def test_legacy_saved_document_json_opens_and_renders(tmp_path):
    """Documents serialized before tone restoration still load and render cleanly."""
    import json

    w, h = 700, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    regions = _add_horizontal_body_lines(img, 100, 800)
    doc = Document(source_path="legacy.pdf", pages=[_make_params(w, h, regions=regions)])
    raw = json.loads(doc.to_json())
    raw.pop("blank_insert_before", None)
    loaded = Document.from_json(json.dumps(raw))
    out_pdf = tmp_path / "legacy_out.pdf"
    build_pdf(loaded, [img], str(out_pdf))
    assert out_pdf.exists() and out_pdf.stat().st_size > 0


def test_wide_flat_gray_background_near_edges_is_restored():
    """Type A: a wide flat gray chapter-cover background touching page edges is restored without vecfill."""
    w, h = 700, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Gray background touching left, right, and bottom edges with a white inner card
    img[140:1000, 0:700] = 190
    img[260:860, 80:620] = 255
    # Dark title text inside the white card (anchors Otsu threshold)
    regions = _add_horizontal_body_lines(img, 420, 620, x0=160, x1=540)
    params = _make_params(w, h, regions=regions)
    settings = RenderSettings(target_dpi=144, output_margin_mm=0.0)

    out_bgr, lines, _, photo_boxes = render_page_image(img, params, settings)
    _, _, tone_alpha, canvas, restore = _prepare_bilevel_layers(
        out_bgr, params, settings, lines, photo_boxes, img.shape
    )
    assert restore.adopted is True
    assert tone_alpha is not None
    # Outer gray background is restored instead of whiting out
    bg_med = float(np.median(canvas[160:220, 100:600, 0]))
    assert 165 <= bg_med <= 215
    # Inner white card body text stays strictly bilevel
    assert set(np.unique(canvas[420:620, 160:540]).tolist()).issubset({0, 255})


def test_small_gray_label_boxes_without_body_lines_are_restored():
    """Type B: small solid gray chapter/section label boxes with knockout text are restored."""
    w, h = 700, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Three adjacent small gray label boxes ("第" "2" "章" style), each ~44x44
    for x0 in (280, 328, 376):
        img[110:155, x0:x0 + 44] = 120
        img[122:143, x0 + 14:x0 + 30] = 252
    # Vertical title ink below (no corroborated body lines)
    for y in range(250, 750, 40):
        img[y:y + 24, 336:362] = 20

    params = _make_params(w, h, regions=[])
    settings = RenderSettings(target_dpi=144, output_margin_mm=0.0)
    out_bgr, lines, _, photo_boxes = render_page_image(img, params, settings)
    _, _, tone_alpha, canvas, restore = _prepare_bilevel_layers(
        out_bgr, params, settings, lines, photo_boxes, img.shape
    )
    assert restore.adopted is True
    assert tone_alpha is not None
    # All three label boxes stay midtone gray rather than crushing to black
    for x0 in (280, 328, 376):
        med = float(np.median(canvas[113:120, x0 + 4:x0 + 40, 0]))
        assert 90 <= med <= 150


def test_thick_gray_decorative_bar_is_restored_while_thin_rule_is_ignored():
    """Type B: a thick gray decorative bar (~14px) is restored, whereas a thin 3px rule is not."""
    w, h = 700, 1000
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Thick gray bar above heading (150x14, tone 150)
    img[200:214, 480:630] = 150
    regions = _add_horizontal_body_lines(img, 320, 820)

    params = _make_params(w, h, regions=regions)
    settings = RenderSettings(target_dpi=144, output_margin_mm=0.0)
    out_bgr, lines, _, photo_boxes = render_page_image(img, params, settings)
    _, _, tone_alpha, canvas, restore = _prepare_bilevel_layers(
        out_bgr, params, settings, lines, photo_boxes, img.shape
    )
    assert restore.adopted is True
    assert tone_alpha is not None
    assert 120 <= float(np.median(canvas[202:212, 500:610, 0])) <= 180

    # Now test a thin 3px horizontal rule alone -> must NOT be restored
    img_rule = np.full((h, w, 3), 255, dtype=np.uint8)
    img_rule[200:203, 200:600] = 150
    regions_rule = _add_horizontal_body_lines(img_rule, 320, 820)
    params_rule = _make_params(w, h, regions=regions_rule)
    out_r, lines_r, _, photo_r = render_page_image(img_rule, params_rule, settings)
    _, _, alpha_r, _, restore_r = _prepare_bilevel_layers(
        out_r, params_rule, settings, lines_r, photo_r, img_rule.shape
    )
    assert restore_r.adopted is False
    assert alpha_r is None
