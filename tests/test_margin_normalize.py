from hokusai_press.geometry.margin import normalize_margins
from hokusai_press.model import Box, Deskew, Margin, PageParams, SourceRef


def _page(idx, content, dpi=600, nombre=None):
    return PageParams(
        source=SourceRef(path="b.pdf", page_index=idx),
        dpi=dpi,
        deskew=Deskew(),
        margin=Margin(content=content, confidence=0.8, nombre_box=nombre),
    )


def test_uniform_size_for_the_bulk_outlier_gets_own_size_no_clip():
    # The bulk of pages share ONE uniform size (an ADF book has one page size).
    # The uniform size is the P97.5 of content extents, so a single large
    # outlier page does NOT inflate every other page's crop; instead the
    # outlier gets its own (content+margin) crop and is never clipped.
    bulk = [
        _page(0, Box(100, 120, 480, 700)),   # ~380 x 580
        _page(1, Box(140, 110, 520, 690)),
        _page(2, Box(90, 130, 470, 710)),
        _page(3, Box(150, 100, 530, 680)),
        _page(5, Box(110, 120, 490, 700)),
        _page(7, Box(120, 110, 500, 690)),
    ]
    outlier = _page(9, Box(60, 80, 700, 1100))   # much larger than the bulk
    pages = bulk + [outlier]
    normalize_margins(pages, output_margin_mm=5.0)

    # the bulk pages all share one size
    w0 = bulk[0].margin.crop.width
    h0 = bulk[0].margin.crop.height
    for p in bulk:
        assert abs(p.margin.crop.width - w0) < 1e-6
        assert abs(p.margin.crop.height - h0) < 1e-6

    # the outlier is NOT clipped and is larger than the uniform size
    assert outlier.margin.crop.width > w0
    assert outlier.margin.crop.height > h0

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


def test_confident_nombre_anchored_horizontally_too():
    # Two confident recto pages whose content boxes differ slightly in WIDTH
    # (real-corpus-scale variation -- a few percent, not a redesigned book)
    # but whose nombre sits the same distance in from the content's left
    # edge. The nombre must land at a near-consistent horizontal offset
    # (small residual jitter is fine; it must NOT systematically drag every
    # page's content off-center the way pure content-edge anchoring did).
    def conf(idx, content, nombre):
        p = _page(idx, content, dpi=600, nombre=nombre)
        p.page_number = 10 + idx
        return p

    pages = [
        conf(0, Box(200, 300, 600, 900), Box(220, 930, 260, 960)),
        conf(2, Box(200, 300, 615, 900), Box(220, 930, 260, 960)),
    ]
    normalize_margins(pages, output_margin_mm=5.0)

    x0 = pages[0].margin.nombre_box.x0 - pages[0].margin.crop.x0
    x1 = pages[1].margin.nombre_box.x0 - pages[1].margin.crop.x0
    assert abs(x0 - x1) < 8.0


def test_nombre_vertical_anchor_is_shared_across_parity():
    # Real corpus finding: nombre.y0 in the RAW scan is ~identical for odd
    # and even pages (a real book prints the nombre at the same height on
    # recto and verso) -- only the horizontal side differs by parity. An
    # earlier version computed the vertical anchor PER PARITY (by analogy
    # with the horizontal one), which let the two parities' centered
    # baselines drift apart and reintroduced an ~80px recto/verso height
    # mismatch that doesn't exist in the source. The vertical anchor must
    # be shared across both parities.
    def conf(idx, content, nombre):
        p = _page(idx, content, dpi=600, nombre=nombre)
        p.page_number = 10 + idx
        return p

    pages = [
        conf(0, Box(200, 300, 600, 900), Box(220, 930, 260, 960)),
        conf(1, Box(250, 280, 650, 920), Box(610, 930, 650, 960)),
        conf(2, Box(200, 300, 600, 900), Box(220, 930, 260, 960)),
        conf(3, Box(250, 280, 650, 920), Box(610, 930, 650, 960)),
    ]
    normalize_margins(pages, output_margin_mm=5.0)

    y0_even = pages[0].margin.nombre_box.y0 - pages[0].margin.crop.y0
    y0_odd = pages[1].margin.nombre_box.y0 - pages[1].margin.crop.y0
    assert abs(y0_even - y0_odd) < 1.0


def test_margin_at_least_output_margin_on_all_sides():
    # the crop must sit >= output margin outside the content on every side
    # (never touching it) -- deterministic centered placement (both axes).
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
        assert c.y0 - crop.y0 >= margin_px - 1e-6      # top
        assert crop.y1 - c.y1 >= margin_px - 1e-6      # bottom
    # each page centered on its own content (the real per-page invariant;
    # uniform-vs-outlier sizing is covered by the dedicated test above)
    for p in pages:
        crop, c = p.margin.crop, p.margin.content
        left_margin = c.x0 - crop.x0
        right_margin = crop.x1 - c.x1
        top_margin = c.y0 - crop.y0
        bottom_margin = crop.y1 - c.y1
        assert abs(left_margin - right_margin) < 1.0   # centered horizontally
        assert abs(top_margin - bottom_margin) < 1.0    # centered vertically


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
