from hokusai_press.geometry.margin import normalize_margins
from hokusai_press.model import Box, Deskew, Margin, PageParams, SourceRef


def _page(idx, content, dpi=600, nombre=None):
    return PageParams(
        source=SourceRef(path="b.pdf", page_index=idx),
        dpi=dpi,
        deskew=Deskew(),
        margin=Margin(content=content, confidence=0.8, nombre_box=nombre),
    )


def test_uniform_size_whole_document_and_no_clip():
    # left (even) and right (odd) pages must end up the SAME size -- an ADF book
    # has one uniform page size, not different sizes per parity.
    pages = [
        _page(0, Box(100, 120, 480, 700)),   # even
        _page(1, Box(140, 110, 520, 690)),   # odd
        _page(2, Box(90, 130, 470, 720)),    # even, tallest content
        _page(3, Box(150, 100, 540, 680)),   # odd, widest content
    ]
    normalize_margins(pages, output_margin_mm=5.0)

    w0 = pages[0].margin.crop.width
    h0 = pages[0].margin.crop.height
    for p in pages:                          # every page identical size
        assert abs(p.margin.crop.width - w0) < 1e-6
        assert abs(p.margin.crop.height - h0) < 1e-6

    # no page's content is clipped by its crop
    for p in pages:
        crop, c = p.margin.crop, p.margin.content
        assert crop.x0 <= c.x0 + 1e-6 and crop.x1 >= c.x1 - 1e-6
        assert crop.y0 <= c.y0 + 1e-6 and crop.y1 >= c.y1 - 1e-6


def test_nombre_anchors_vertical_position():
    # two pages, same content height, nombre at different absolute y; after
    # normalization the nombre should sit at the same offset from the crop top
    pages = [
        _page(0, Box(100, 200, 480, 800), nombre=Box(280, 860, 300, 880)),
        _page(2, Box(100, 240, 480, 840), nombre=Box(280, 900, 300, 920)),
    ]
    normalize_margins(pages, output_margin_mm=5.0)
    off0 = pages[0].margin.nombre_box.y0 - pages[0].margin.crop.y0
    off1 = pages[1].margin.nombre_box.y0 - pages[1].margin.crop.y0
    assert abs(off0 - off1) < 1e-6


def test_mixed_dpi_yields_equal_physical_size():
    # same physical content (2 inch wide) at different dpi -> equal output inches
    pages = [
        _page(0, Box(0, 0, 1200, 1800), dpi=600),  # 2.0 x 3.0 inch
        _page(2, Box(0, 0, 600, 900), dpi=300),    # 2.0 x 3.0 inch
    ]
    normalize_margins(pages, output_margin_mm=5.0)
    in0 = pages[0].margin.crop.width / 600
    in1 = pages[1].margin.crop.width / 300
    assert abs(in0 - in1) < 1e-6
