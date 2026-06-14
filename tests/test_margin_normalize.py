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


def test_bad_nombre_does_not_clip_content():
    # a wildly wrong nombre (far below the content) must NOT shift the crop off
    # the content -- the no-clip clamp keeps the whole body inside the crop.
    pages = [
        _page(0, Box(100, 100, 500, 900), nombre=Box(280, 60, 300, 80)),
        _page(2, Box(100, 100, 500, 900), nombre=Box(280, 1500, 300, 1520)),
    ]
    normalize_margins(pages, output_margin_mm=5.0)
    for p in pages:
        crop, c = p.margin.crop, p.margin.content
        assert crop.x0 <= c.x0 + 1e-6 and crop.x1 >= c.x1 - 1e-6
        assert crop.y0 <= c.y0 + 1e-6 and crop.y1 >= c.y1 - 1e-6


def test_confident_nombre_anchored_else_fallback_all_keep_margin():
    margin_px = 5.0 / 25.4 * 600

    def conf(idx, content, nombre):
        p = _page(idx, content, dpi=600, nombre=nombre)
        p.page_number = 10 + idx           # mark as confidently numbered
        return p

    # two confident recto pages: nombre sits the same distance below content top
    pages = [
        conf(0, Box(200, 300, 600, 900), Box(380, 930, 420, 960)),
        conf(2, Box(200, 350, 600, 950), Box(380, 980, 420, 1010)),
        _page(4, Box(150, 200, 650, 1150)),   # tallest, low-confidence (no number)
    ]
    normalize_margins(pages, output_margin_mm=5.0)

    for p in pages:                           # >= margin on every side, always
        crop, c = p.margin.crop, p.margin.content
        assert c.x0 - crop.x0 >= margin_px - 1e-6
        assert crop.x1 - c.x1 >= margin_px - 1e-6
        assert c.y0 - crop.y0 >= margin_px - 1e-6
        assert crop.y1 - c.y1 >= margin_px - 1e-6

    # confident pages: the nombre lands at a consistent offset from the crop top
    o0 = pages[0].margin.nombre_box.y0 - pages[0].margin.crop.y0
    o1 = pages[1].margin.nombre_box.y0 - pages[1].margin.crop.y0
    assert abs(o0 - o1) < 1.0


def test_margin_at_least_output_margin_on_all_sides():
    # the crop must sit >= output margin outside the content on every side
    # (never touching it) -- deterministic centered + head-margin placement.
    dpi = 600
    margin_px = 5.0 / 25.4 * dpi
    pages = [
        _page(0, Box(200, 200, 600, 1000), dpi=dpi),
        _page(1, Box(180, 220, 560, 980), dpi=dpi),   # smaller content
    ]
    normalize_margins(pages, output_margin_mm=5.0)
    for p in pages:
        crop, c = p.margin.crop, p.margin.content
        assert c.x0 - crop.x0 >= margin_px - 1e-6      # left
        assert crop.x1 - c.x1 >= margin_px - 1e-6      # right
        assert c.y0 - crop.y0 >= margin_px - 1e-6      # top (head)
        assert crop.y1 - c.y1 >= margin_px - 1e-6      # bottom
    # constant head margin across pages
    head0 = pages[0].margin.content.y0 - pages[0].margin.crop.y0
    head1 = pages[1].margin.content.y0 - pages[1].margin.crop.y0
    assert abs(head0 - head1) < 1e-6


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
