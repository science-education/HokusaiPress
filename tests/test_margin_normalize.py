from hokusai_press.geometry.margin import normalize_margins
from hokusai_press.model import Box, Deskew, Margin, PageParams, SourceRef


def _page(idx, content, dpi=600, nombre=None):
    return PageParams(
        source=SourceRef(path="b.pdf", page_index=idx),
        dpi=dpi,
        deskew=Deskew(),
        margin=Margin(content=content, confidence=0.8, nombre_box=nombre),
    )


def _numbered_page(idx, content, page_w, page_h, nombre, dpi=600):
    """A page with page_w/page_h set -- only pages with this populated (real
    analyze_document output, not the bare _page() fixture above) feed the
    robust per-side margin statistic in normalize_margins."""
    p = _page(idx, content, dpi=dpi, nombre=nombre)
    p.margin.page_w, p.margin.page_h = page_w, page_h
    return p


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


def test_body_centered_symmetric_left_right_via_nombre():
    # No 柱 (no regions) so body box == content box. recto fore-edge = right
    # (nombre at x~665), verso fore-edge = left (nombre at x~335). The body
    # must end up horizontally CENTERED -- left margin == right margin on
    # every page -- with one uniform crop width, regardless of the raw
    # binding-side asymmetry (recto left gap 100 vs right 300, etc.).
    pages = [
        _numbered_page(0, Box(100, 250, 700, 1300), 1000, 1400,
                       Box(650, 1305, 680, 1325)),
        _numbered_page(1, Box(300, 250, 900, 1300), 1000, 1400,
                       Box(320, 1305, 350, 1325)),
        _numbered_page(2, Box(100, 250, 700, 1300), 1000, 1400,
                       Box(650, 1305, 680, 1325)),
        _numbered_page(3, Box(300, 250, 900, 1300), 1000, 1400,
                       Box(320, 1305, 350, 1325)),
    ]
    normalize_margins(pages, output_margin_mm=5.0)

    # crop width = body_w(600) + 2*side_margin, side_margin = mean of the L/R
    # body margins ((100+300)/2 = 200) -> 600 + 400 = 1000.
    for p in pages:
        assert abs(p.margin.crop.width - 1000.0) < 1.0
    widths = {round(p.margin.crop.width, 1) for p in pages}
    assert len(widths) == 1
    # the actual goal: each page's left margin equals its right margin
    # (clean centering -- what perfect deskew makes conspicuous if broken).
    for p in pages:
        c, crop = p.margin.content, p.margin.crop
        assert abs((c.x0 - crop.x0) - (crop.x1 - c.x1)) < 1.0


def test_robust_margin_keeps_top_bottom_asymmetric():
    # top gap (250px) and bottom gap (100px) are NOT a binding artifact (no
    # left/right flip with parity), so they must stay asymmetric -- unlike
    # left/right, there is no single "fore-edge" value to fall back on here.
    pages = [
        _numbered_page(0, Box(100, 250, 700, 1300), 1000, 1400,
                       Box(650, 1305, 680, 1325)),
        _numbered_page(1, Box(300, 250, 900, 1300), 1000, 1400,
                       Box(320, 1305, 350, 1325)),
        _numbered_page(2, Box(100, 250, 700, 1300), 1000, 1400,
                       Box(650, 1305, 680, 1325)),
        _numbered_page(3, Box(300, 250, 900, 1300), 1000, 1400,
                       Box(320, 1305, 350, 1325)),
    ]
    normalize_margins(pages, output_margin_mm=5.0)

    # body + top + bottom; a paper margin below the 5mm output margin is
    # raised to it (100px -> 118px at 600dpi)
    expected_h = 1050.0 + 250.0 + max(100.0, 5.0 / 25.4 * 600)
    for p in pages:
        assert abs(p.margin.crop.height - expected_h) < 1.0


def test_divider_page_without_nombre_keeps_proportional_position():
    # numbered pages (with page_w/h + nombre) establish the robust-margin
    # statistic; a divider/cover page WITHOUT a nombre must still use the
    # old proportional baseline() -- its content sits in the upper-left of
    # the original page (fx, fy both small), and that relative position
    # must survive onto the new uniform-size page rather than being pulled
    # to the numbered pages' fore-edge/top/bottom margin scheme.
    numbered = [
        _numbered_page(0, Box(100, 250, 700, 1300), 1000, 1400,
                       Box(650, 1305, 680, 1325)),
        _numbered_page(1, Box(300, 250, 900, 1300), 1000, 1400,
                       Box(320, 1305, 350, 1325)),
        _numbered_page(2, Box(100, 250, 700, 1300), 1000, 1400,
                       Box(650, 1305, 680, 1325)),
        _numbered_page(3, Box(300, 250, 900, 1300), 1000, 1400,
                       Box(320, 1305, 350, 1325)),
    ]
    # off-centre but not glued to the paper edge (>5mm in on every side, so
    # the no-clip margin floor doesn't have to move it)
    divider_content = Box(150, 250, 450, 600)
    divider = _page(4, divider_content, dpi=600, nombre=None)
    divider.margin.page_w, divider.margin.page_h = 1000.0, 1400.0
    pages = numbered + [divider]

    normalize_margins(pages, output_margin_mm=5.0)

    c = divider.margin.content
    crop = divider.margin.crop
    orig_fx = (c.x0 + c.width / 2) / divider.margin.page_w
    orig_fy = (c.y0 + c.height / 2) / divider.margin.page_h
    new_cx = (c.x0 + c.width / 2 - crop.x0) / crop.width
    new_cy = (c.y0 + c.height / 2 - crop.y0) / crop.height
    assert abs(orig_fx - new_cx) < 0.05
    assert abs(orig_fy - new_cy) < 0.05


def _text(box):
    from hokusai_press.model import Region, RegionKind
    return Region(kind=RegionKind.TEXT, box=box)


def _tategaki_page(idx, dx, dy, ncols=10, page_w=1400, page_h=2200, extra=()):
    """Columns 40px wide, 76px pitch, 1600px tall; the type area's top-right
    corner at (1200+dx, 300+dy). Page number far from the body (bottom)."""
    cols = [Box(1160 + dx - 76 * k, 300 + dy, 1200 + dx - 76 * k, 1900 + dy) for k in range(ncols)]
    regions = [_text(b) for b in cols] + [_text(b) for b in extra]
    xs = [b.x0 for b in cols] + [b.x0 for b in extra]
    content = Box(min(xs), 300 + dy, 1200 + dx, 1900 + dy)
    p = _numbered_page(idx, content, page_w, page_h, None, dpi=300)
    p.regions = regions
    return p


def test_tategaki_pages_register_on_the_type_area_start_corner():
    # scan shift differs page to page (a few mm); short last page of a
    # chapter (3 columns) and a page with a stray mark as "nombre" too
    shifts = [(0, 0), (30, -20), (-25, 15), (10, 25), (-15, -10), (20, 5)]
    pages = [_tategaki_page(i, dx, dy) for i, (dx, dy) in enumerate(shifts)]
    pages.append(_tategaki_page(6, 12, -8, ncols=3))
    pages[2].margin.nombre_box = Box(1007, 162, 1011, 166)   # a speck read as "9"
    normalize_margins(pages, output_margin_mm=5.0)
    pos = {(round(p.margin.crop.x1 - (1200 + d[0]), 3), round((300 + d[1]) - p.margin.crop.y0, 3))
           for p, d in zip(pages, shifts + [(12, -8)])}
    assert len(pos) == 1          # start corner at one output position on every page
    w = {round(p.margin.crop.width, 3) for p in pages}
    assert len(w) == 1
    # equal left/right margins around the full type area
    p = pages[0]
    assert abs((p.margin.content.x0 - p.margin.crop.x0) - (p.margin.crop.x1 - p.margin.content.x1)) < 1.0


def test_part_title_page_is_not_registered_on_the_type_area():
    body = [_tategaki_page(i, 0, 0) for i in range(6)]
    title = _numbered_page(6, Box(650, 800, 700, 1300), 1400, 2200, None, dpi=300)
    title.regions = [_text(Box(650, 800, 700, 1300))]    # one short line, mid-page
    normalize_margins(body + [title], output_margin_mm=5.0)
    crop, c = title.margin.crop, title.margin.content
    # keeps its proportional place on the paper (centre ~0.48, ~0.48)
    assert abs(((c.x0 + c.x1) / 2 - crop.x0) / crop.width - 675 / 1400) < 0.01
    assert abs(((c.y0 + c.y1) / 2 - crop.y0) / crop.height - 1050 / 2200) < 0.01


def _yokogaki_page(idx, lines):
    p = _numbered_page(idx, Box(min(b.x0 for b in lines), min(b.y0 for b in lines),
                                max(b.x1 for b in lines), max(b.y1 for b in lines)),
                       1400, 2200, None, dpi=300)
    p.regions = [_text(b) for b in lines]
    return p


def test_minority_direction_pages_without_consensus_are_not_registered():
    # img20261121_0015: a tategaki book whose cover and colophon are the only
    # yokogaki pages. Two pages are no type area: the cover keeps its place.
    body = [_tategaki_page(i, 0, 0) for i in range(1, 12) if i != 2]
    cover = _yokogaki_page(0, [Box(400, 700 + 90 * k, 1000, 760 + 90 * k) for k in range(5)])
    colophon = _yokogaki_page(2, [Box(150, 1500 + 60 * k, 900, 1540 + 60 * k) for k in range(8)])
    normalize_margins([cover, colophon] + body, output_margin_mm=5.0)
    crop, c = cover.margin.crop, cover.margin.content
    assert abs(((c.x0 + c.x1) / 2 - crop.x0) / crop.width - 700 / 1400) < 0.01
    assert abs(((c.y0 + c.y1) / 2 - crop.y0) / crop.height - c.y0 / 2200 - (c.height / 2) / 2200) < 0.01


def test_full_bleed_sheet_larger_than_the_page_is_trimmed_not_shrunk():
    # img20261121_0010 p15: a black chapter-title page scanned past the trim
    # (4% wider than the book's page, its top and bottom cut by the scan).
    # Shrunk to fit it left white bands; the page is a trim-size window of it.
    body = [_tategaki_page(i, 0, 0) for i in range(1, 9)]
    normalize_margins(body, output_margin_mm=5.0)
    tw, th = body[0].margin.target_w, body[0].margin.target_h
    black = _numbered_page(9, Box(0, 0, tw * 1.04, th * 0.98), tw * 1.04, th * 0.98, None, dpi=300)
    black.regions = [_text(Box(tw * 0.4, th * 0.1, tw * 0.45, th * 0.4))]
    pages = body + [black]
    normalize_margins(pages, output_margin_mm=5.0)
    m = black.margin
    assert m.fill
    assert abs(m.crop.width / m.crop.height - tw / th) < 1e-6        # the page's shape
    assert m.crop.height <= m.content.height + 1e-6                  # all sheet, no white
    assert m.crop.width <= m.content.width + 1e-6
    assert abs((m.crop.x0 + m.crop.x1) - (m.content.x0 + m.content.x1)) < 1e-6   # centred
    from hokusai_press.model import Flag
    assert Flag.CONTENT_SCALED_DOWN not in black.flags
    assert not any(p.margin.fill for p in body)


def test_full_bleed_sheet_far_too_large_is_still_shrunk():
    body = [_tategaki_page(i, 0, 0) for i in range(1, 9)]
    normalize_margins(body, output_margin_mm=5.0)
    tw, th = body[0].margin.target_w, body[0].margin.target_h
    big = _numbered_page(9, Box(0, 0, tw * 1.6, th * 1.6), tw * 1.6, th * 1.6, None, dpi=300)
    normalize_margins(body + [big], output_margin_mm=5.0)
    assert not big.margin.fill


def test_image_without_dpi_keeps_requested_margin_mm_at_render():
    """A11: When dpi is None (e.g. PNG/JPEG without DPI metadata), normalize_margins
    and render.compose_transform must use the same DPI assumption so 5mm stays ~5mm."""
    import pytest
    from hokusai_press.model import RenderSettings
    from hokusai_press.render import compose_transform

    page = _numbered_page(0, Box(10, 10, 610, 1010), 620, 1020, None, dpi=None)
    normalize_margins([page], output_margin_mm=5.0)
    settings = RenderSettings(target_dpi=600, output_margin_mm=5.0)
    _, (out_w, out_h), _ = compose_transform((1020, 620), page, settings)
    # Content is 600x1000 at 600 dpi -> margin on each side should be ~5mm (not ~0.8mm)
    margin_x_mm = ((out_w - 600) / 2) / settings.target_dpi * 25.4
    margin_y_mm = ((out_h - 1000) / 2) / settings.target_dpi * 25.4
    assert margin_x_mm == pytest.approx(5.0, abs=0.2)
    assert margin_y_mm == pytest.approx(5.0, abs=0.2)



def _crop_page(idx, cx_off, cy_off, number=None, src=None, regions=()):
    # dpi 254: 1 px = 0.1 mm; a 1000 x 1500 sheet, a 900 x 1400 crop
    p = _numbered_page(idx, Box(100, 100, 900, 1400), 1000, 1500, None, dpi=254)
    x0, y0 = 500 + cx_off - 450, 750 + cy_off - 700
    p.margin.crop = Box(x0, y0, x0 + 900, y0 + 1400)
    p.page_number, p.page_number_source = number, src
    p.regions = list(regions)
    return p


def test_matter_page_gets_the_parity_correction_and_its_sheets_other_side():
    from hokusai_press.geometry.margin import place_matter_pages
    base = {0: (40.0, 5.0), 1: (-40.0, 5.0)}
    pages = []
    for i in range(1, 25):
        k = i // 2                                   # sheet (2k, 2k+1)
        rx, ry = ((k % 3) - 1) * 6.0, ((k % 3) - 1) * 4.0
        sx = rx if i % 2 == 0 else -rx               # turned over: mirrored
        pages.append(_crop_page(i, base[i % 2][0] + sx, base[i % 2][1] + ry, i, "folio"))
    cover = _crop_page(0, 300.0, 300.0)              # wherever Pass 1 put it
    colophon = _crop_page(25, -300.0, 0.0)
    pages = [cover] + pages + [colophon]
    pass1 = {"matter_ids": {id(cover), id(colophon)}, "use_dpi": True, "fallback_dpi": 600.0}
    assert place_matter_pages(pages, pass1) == 2

    def off(p):
        return ((p.margin.crop.x0 + p.margin.crop.x1) / 2 - 500,
                (p.margin.crop.y0 + p.margin.crop.y1) / 2 - 750)
    # cover: page 1 (other side) deviates by (+6, -4) from its parity median
    assert abs(off(cover)[0] - (40.0 - 6.0)) < 0.5
    assert abs(off(cover)[1] - (5.0 - 4.0)) < 0.5
    # colophon: page 24 (sheet 12, k%3=0) deviates by (-6, -4)
    assert abs(off(colophon)[0] - (-40.0 + 6.0)) < 0.5
    assert abs(off(colophon)[1] - (5.0 - 4.0)) < 0.5


def test_matter_page_never_clips_characters():
    from hokusai_press.geometry.margin import place_matter_pages
    pages = [_crop_page(i, (60.0 if i % 2 == 0 else -60.0), 0.0, i, "folio") for i in range(1, 25)]
    # a colophon whose text runs to x=960, 4 mm from the sheet edge (the crop
    # at the odd-parity correction -60 would end at 890 and clip it)
    col = _crop_page(25, 0.0, 0.0, regions=[_text(Box(600, 1200, 960, 1300))])
    pages.append(col)
    place_matter_pages(pages, {"matter_ids": {id(col)}, "use_dpi": True, "fallback_dpi": 600.0})
    assert col.margin.crop.x1 >= 960
