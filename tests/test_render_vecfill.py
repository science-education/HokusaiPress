import numpy as np
import pytest

pytest.importorskip("hybrid_ocr")
pytest.importorskip("pikepdf")

import pikepdf

from hokusai_press.model import (
    Box,
    Deskew,
    Document,
    Margin,
    PageParams,
    Region,
    RegionKind,
    RenderSettings,
    SourceRef,
)
from hokusai_press.mrc import MrcPageBuilder
from hokusai_press.render import build_pdf, render_page_image, _set_physical_page_size


def _params(regions):
    return PageParams(
        source=SourceRef(path="synthetic.png"),
        dpi=600,
        deskew=Deskew(angle_deg=0.0, confidence=5.0),
        margin=Margin(content=Box(0, 0, 360, 480), confidence=0.9),
        regions=regions,
    )


def _text_photo_page():
    img = np.full((480, 360, 3), 255, dtype=np.uint8)
    for y in range(40, 180, 28):
        img[y:y + 10, 40:300] = 0
    grad = np.tile(np.linspace(40, 220, 220, dtype=np.uint8), (160, 1))
    img[260:420, 70:290] = np.dstack([grad, grad, grad])
    regions = [
        Region(kind=RegionKind.TEXT, box=Box(40, 40, 300, 50), ocr_text="sample"),
        Region(kind=RegionKind.PHOTO, box=Box(70, 260, 290, 420), tone="gray"),
    ]
    return img, _params(regions)


def _flat_figure_page():
    img = np.full((480, 360, 3), 255, dtype=np.uint8)
    img[140:340, 70:290] = 170
    regions = [Region(kind=RegionKind.FIGURE, box=Box(70, 140, 290, 340))]
    return img, _params(regions)


def _old_render_pdf(img, params, out_path):
    settings = RenderSettings(target_dpi=600, output_margin_mm=0.0)
    out_bgr, lines, mode, photo_boxes = render_page_image(img, params, settings)
    builder = MrcPageBuilder(
        compress=settings.bilevel_codec,
        target_dpi=settings.target_dpi,
        jpeg_quality=settings.jpeg_quality,
    )
    builder.add_page(out_bgr, lines, photo_boxes, mode)
    builder.save(str(out_path))
    _set_physical_page_size(str(out_path), settings.target_dpi)


def _optimized_pdf(img, params, out_path):
    doc = Document(
        source_path="synthetic.png",
        pages=[params],
        render=RenderSettings(target_dpi=600, output_margin_mm=0.0),
    )
    build_pdf(doc, [img], str(out_path))


def _raster_baseline_pdf(img, params, out_path):
    settings = RenderSettings(target_dpi=600, output_margin_mm=0.0)
    out_bgr, lines, mode, _ = render_page_image(img, params, settings)
    builder = MrcPageBuilder(
        compress=settings.bilevel_codec,
        target_dpi=settings.target_dpi,
        jpeg_quality=settings.jpeg_quality,
    )
    builder.add_page(out_bgr, lines, [(70, 140, 290, 340, "gray")], mode)
    builder.save(str(out_path))
    _set_physical_page_size(str(out_path), settings.target_dpi)


def _content_bytes(path):
    with pikepdf.open(path) as pdf:
        contents = pdf.pages[0].Contents
        if isinstance(contents, pikepdf.Array):
            return b"\n".join(stream.read_bytes() for stream in contents)
        return contents.read_bytes()


def test_m1_no_solid_fill_page_is_byte_identical_to_old_render_path(tmp_path):
    img, params = _text_photo_page()
    baseline = tmp_path / "baseline.pdf"
    optimized = tmp_path / "optimized.pdf"

    _old_render_pdf(img, params, baseline)
    _optimized_pdf(img, params, optimized)

    assert optimized.read_bytes() == baseline.read_bytes()


def test_m2_flat_figure_vecfill_is_smaller_than_raster_baseline(tmp_path):
    img, params = _flat_figure_page()
    raster = tmp_path / "raster.pdf"
    optimized = tmp_path / "optimized.pdf"

    _raster_baseline_pdf(img, params, raster)
    _optimized_pdf(img, params, optimized)

    assert optimized.stat().st_size < raster.stat().st_size


def test_m3_flat_figure_content_stream_contains_vecfill_ops(tmp_path):
    img, params = _flat_figure_page()
    optimized = tmp_path / "optimized.pdf"

    _optimized_pdf(img, params, optimized)
    content = _content_bytes(optimized)

    assert b" re " in content
    assert b" f" in content
    assert b" g " in content
