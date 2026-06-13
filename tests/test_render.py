import numpy as np

from hokusai_press.model import (
    Box,
    Deskew,
    Margin,
    PageKind,
    PageParams,
    Region,
    RegionKind,
    RenderSettings,
    SourceRef,
)
from hokusai_press.render import (
    compose_transform,
    render_output_preview,
    render_page_image,
    _map_box,
)


def _params(margin_box, dpi=600, kind=PageKind.AUTO, regions=None):
    return PageParams(
        source=SourceRef(path="x.png"),
        dpi=dpi,
        deskew=Deskew(angle_deg=0.0, confidence=5.0),
        margin=Margin(content=margin_box, confidence=0.9),
        regions=regions or [],
        page_kind=kind,
    )


def test_compose_crops_and_scales_identity():
    settings = RenderSettings(target_dpi=600, output_margin_mm=0.0)
    params = _params(Box(50, 100, 250, 500), dpi=600)
    M, out_size, scale = compose_transform((800, 600), params, settings)
    assert scale == 1.0
    assert out_size == (200, 400)
    mapped = _map_box(Box(50, 100, 250, 500), M)
    assert abs(mapped.x0) < 1e-6 and abs(mapped.y0) < 1e-6
    assert abs(mapped.x1 - 200) < 1e-6 and abs(mapped.y1 - 400) < 1e-6


def test_compose_upscales_to_target_dpi():
    settings = RenderSettings(target_dpi=600, output_margin_mm=0.0)
    params = _params(Box(0, 0, 100, 100), dpi=300)
    M, out_size, scale = compose_transform((200, 200), params, settings)
    assert scale == 2.0
    assert out_size == (200, 200)


def test_render_one_shot_output_shape_and_mode():
    settings = RenderSettings(target_dpi=600, output_margin_mm=0.0)
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    regions = [Region(kind=RegionKind.TEXT, box=Box(60, 110, 240, 130),
                      ocr_text="あいう")]
    params = _params(Box(50, 100, 250, 500), regions=regions)
    out, lines, mode, photo_boxes = render_page_image(img, params, settings)
    assert out.shape[0] == 400 and out.shape[1] == 200
    assert mode == "bw"  # no photo region -> bilevel
    assert photo_boxes == []
    assert lines and lines[0]["text"] == "あいう"
    # text box mapped into output space (origin shifted by crop)
    assert lines[0]["box"][0] >= 0 and lines[0]["box"][1] >= 0


def test_photo_region_becomes_overlay_box():
    settings = RenderSettings(output_margin_mm=0.0)
    regions = [Region(kind=RegionKind.PHOTO, box=Box(60, 200, 240, 400))]
    params = _params(Box(50, 100, 250, 500), regions=regions)
    _, _, mode, photo_boxes = render_page_image(
        np.full((800, 600, 3), 255, np.uint8), params, settings)
    assert mode == "bw"  # MRC: bilevel base, photo as overlay
    assert len(photo_boxes) == 1
    # mapped into output space: original (60,200)-(240,400) shifted by crop origin
    x0, y0, x1, y1, tone = photo_boxes[0]
    assert abs(x0 - 10) < 1e-6 and abs(y0 - 100) < 1e-6
    assert abs(x1 - 190) < 1e-6 and abs(y1 - 300) < 1e-6
    assert tone is None  # no override -> auto gray/color at encode time


def test_output_preview_bw_is_binarized():
    # a gray page with no photo region -> bw preview is pure {0,255} per channel
    settings = RenderSettings(output_margin_mm=0.0)
    img = np.full((400, 300, 3), 128, dtype=np.uint8)
    img[20:40, 20:280] = 30   # a dark bar
    params = _params(Box(0, 0, 300, 400), dpi=600, kind=PageKind.BW)
    prev = render_output_preview(img, params, settings)
    assert prev.ndim == 3
    assert set(np.unique(prev)).issubset({0, 255})  # binarized


def test_output_preview_keeps_photo_region_gray():
    # bw page with a colored photo region forced to gray: that region must carry
    # mid-tones (not pure black/white) and be colorless (B==G==R).
    settings = RenderSettings(output_margin_mm=0.0)
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    img[100:300, 50:250] = (200, 120, 40)  # a flat colored block
    regions = [Region(kind=RegionKind.PHOTO, box=Box(50, 100, 250, 300),
                      tone="gray")]
    params = _params(Box(0, 0, 300, 400), dpi=600, kind=PageKind.BW,
                     regions=regions)
    prev = render_output_preview(img, params, settings)
    patch = prev[150:250, 100:200]
    assert np.any((patch > 0) & (patch < 255))            # has mid-tones (gray)
    assert np.allclose(patch[..., 0], patch[..., 1]) and \
           np.allclose(patch[..., 1], patch[..., 2])      # grayscale (B==G==R)


def test_output_preview_color_page_stays_color():
    settings = RenderSettings(output_margin_mm=0.0)
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    img[100:300, 50:250] = (200, 120, 40)
    params = _params(Box(0, 0, 300, 400), dpi=600, kind=PageKind.COLOR)
    prev = render_output_preview(img, params, settings)
    patch = prev[150:250, 100:200]
    assert not np.allclose(patch[..., 0], patch[..., 2])   # color preserved


def test_photo_region_tone_override_is_carried():
    settings = RenderSettings(output_margin_mm=0.0)
    regions = [Region(kind=RegionKind.PHOTO, box=Box(60, 200, 240, 400),
                      tone="gray")]
    params = _params(Box(50, 100, 250, 500), regions=regions)
    _, _, _, photo_boxes = render_page_image(
        np.full((800, 600, 3), 255, np.uint8), params, settings)
    assert photo_boxes[0][4] == "gray"
