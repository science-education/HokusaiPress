import cv2
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
from hokusai_press.geometry.margin import remove_edge_shadows
from hokusai_press.render import (
    compose_transform,
    render_output_preview,
    render_page_image,
    _map_box,
)


def test_blank_page_renders_white():
    # a page marked blank (OCR found no content + no ink) renders pure white,
    # even if the source still has faint show-through marks
    settings = RenderSettings(target_dpi=600, output_margin_mm=0.0)
    img = np.full((800, 600, 3), 250, dtype=np.uint8)
    img[300:340, 200:400] = 205          # faint show-through (never real ink)
    params = _params(Box(0, 0, 600, 800), dpi=600)
    params.blank = True
    out, lines, mode, photos = render_page_image(img, params, settings)
    assert (out == 255).all() and lines == [] and photos == []


def test_non_blank_page_is_not_whitened():
    settings = RenderSettings(target_dpi=600, output_margin_mm=0.0)
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    img[100:500, 120:240] = 0
    params = _params(Box(100, 100, 250, 500), dpi=600)   # blank defaults False
    out, _, _, _ = render_page_image(img, params, settings)
    assert (out == 0).any()              # content kept


def test_margin_fill_whitens_outside_content_keeps_content():
    # a uniform-size crop is larger than this page's content, pulling in an edge
    # line that sits OUTSIDE the content box. margin fill must whiten it while
    # keeping the content.
    settings = RenderSettings(target_dpi=600, output_margin_mm=0.0)
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    img[200:550, 220:380] = 0      # body content (inside content box)
    img[200:550, 130:140] = 0      # dark edge line, left of content
    params = _params(Box(200, 150, 400, 560), dpi=600)
    params.margin.crop = Box(100, 100, 500, 650)   # crop wider than content
    out, _, _, _ = render_page_image(img, params, settings)
    # crop x0=100, scale 1 -> output x = source_x - 100. content body at out
    # x120-280; the edge line at out x30-40 is in the left margin -> whitened.
    assert (out == 0).any()                    # content kept
    assert (out[:, 0:110] == 0).sum() == 0     # left margin (edge line) whitened


def test_remove_edge_shadows_whitens_band_keeps_content():
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    img[:, 294:300] = 0          # full-height dark bar at the right edge = shadow
    img[120:300, 90:200] = 0     # interior content block (touches no edge)
    out = remove_edge_shadows(img)
    assert (out[:, 294:300] == 255).all()      # shadow removed
    assert (out[150:250, 110:180] == 0).all()  # interior content preserved


def test_remove_edge_shadows_keeps_short_edge_mark():
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    img[10:40, 295:300] = 0      # short edge mark (< half the side) = not a shadow
    out = remove_edge_shadows(img)
    assert (out[10:40, 295:300] == 0).all()    # kept (too short to be a shadow)


def test_remove_edge_shadows_removes_partial_height_extreme_edge_line():
    # Real-world finding (img20260427_0001.pdf, 0427_0010, 0430_0002): a 1-6px
    # line hugging the SAME x just inside the raw scan edge, covering only
    # ~15-50% of the page height (partial scanner/ADF contact) -- too short
    # for the >=50%-of-side shadow rule, but no real text ever starts this
    # close (1.8%) to the physical edge, so a thin extreme-edge line is a
    # shadow fragment regardless of how much of the side it covers.
    img = np.full((1000, 700, 3), 255, dtype=np.uint8)
    img[400:650, 10:13] = 0      # 3px wide, 25% tall, x=10 is 1.4% of width
    img[300:700, 90:250] = 0     # interior body-text block, untouched
    out = remove_edge_shadows(img)
    assert (out[400:650, 10:13] == 255).all()   # fragment removed
    assert (out[300:700, 90:250] == 0).all()    # real content preserved


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


def test_map_box_under_rotation_covers_all_four_corners():
    # Real-corpus finding (img20260427_0001 p11): with a deskew rotation in
    # M, mapping only the box's top-left/bottom-right corners under-covers
    # the box for a far corner (e.g. a text line near the very top of a
    # tall page, distant from the rotation center) -- the margin-fill step
    # then whitens part of real text past the wrongly-short edge it computes
    # from those 2 corners. A small rotation must still produce a bounding
    # box that fully contains all 4 transformed corners.
    cx, cy = 100.0, 100.0
    R = cv2.getRotationMatrix2D((cx, cy), -0.4, 1.0)
    M = R.astype(np.float32)
    box = Box(10, 10, 190, 190)
    pts = np.array([[box.x0, box.y0, 1], [box.x1, box.y0, 1],
                     [box.x0, box.y1, 1], [box.x1, box.y1, 1]], dtype=np.float64).T
    true_corners = (M.astype(np.float64) @ pts)
    mapped = _map_box(box, M)
    assert mapped.x0 <= true_corners[0].min() + 1e-6
    assert mapped.x1 >= true_corners[0].max() - 1e-6
    assert mapped.y0 <= true_corners[1].min() + 1e-6
    assert mapped.y1 >= true_corners[1].max() - 1e-6


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
