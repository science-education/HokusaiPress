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

    expected_h = 1050.0 + 250.0 + 100.0   # uni_h(content) + top + bottom
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


def _with_regions(p, boxes):
    from hokusai_press.model import Region, RegionKind

    p.regions = [Region(kind=RegionKind.TEXT, box=b) for b in boxes]
    return p


def test_full_pages_body_centered_left_right():
    # Printing-plate model (user decision 2026-07-05): on FULL pages the body
    # hull ~= the plate, so landing the nombre on the standard point centers
    # the body left-right.
    pages = [
        _with_regions(
            _numbered_page(i, Box(200, 300, 800, 1300), 1000, 1400,
                           Box(750, 1305, 780, 1325)),
            [Box(200, 300, 800, 1290)])
        for i in (0, 2, 4, 6)
    ]
    normalize_margins(pages, output_margin_mm=5.0)

    for p in pages:
        from hokusai_press.geometry.margin import body_text_box
        bb = body_text_box(p)
        crop = p.margin.crop
        left = bb.x0 - crop.x0
        right = crop.x1 - bb.x1
        assert abs(left - right) < 2.0, (p.source.page_index, left, right)


def test_chapter_end_partial_page_restores_plate_not_hull_center():
    # A tategaki chapter-end page: only the fore-edge half of the plate has
    # ink, the nombre stays at its plate spot. The crop must align with the
    # full pages' crop (plate restored) -- NOT center the shrunken hull.
    full = [
        _with_regions(
            _numbered_page(i, Box(200, 300, 800, 1300), 1000, 1400,
                           Box(750, 1305, 780, 1325)),
            [Box(200, 300, 800, 1290)])
        for i in (0, 2, 4)
    ]
    partial = _with_regions(
        _numbered_page(6, Box(500, 300, 800, 1300), 1000, 1400,
                       Box(750, 1305, 780, 1325)),   # same plate spot
        [Box(500, 300, 800, 1290)])
    pages = full + [partial]
    normalize_margins(pages, output_margin_mm=5.0)

    # identical page coords + identical nombre spot -> identical crop x
    assert abs(partial.margin.crop.x0 - full[0].margin.crop.x0) < 2.0
    # and the hull is deliberately NOT centered (text stays at the fore edge)
    bb = Box(500, 300, 800, 1290)
    left = bb.x0 - partial.margin.crop.x0
    right = partial.margin.crop.x1 - bb.x1
    assert left - right > 100


def test_atypical_nombre_falls_back_to_hull_centering():
    # A "nombre" detected far from the parity's typical spot is a false
    # positive; anchoring on it would fling the page. Fall back to centering.
    pages = [
        _with_regions(
            _numbered_page(i, Box(200, 300, 800, 1300), 1000, 1400,
                           Box(750, 1305, 780, 1325)),
            [Box(200, 300, 800, 1290)])
        for i in (0, 2, 4)
    ] + [
        _with_regions(
            _numbered_page(6, Box(200, 300, 800, 1300), 1000, 1400,
                           Box(210, 700, 240, 720)),   # mid-page "nombre"
            [Box(200, 300, 800, 1290)]),
    ]
    normalize_margins(pages, output_margin_mm=5.0)

    p = pages[-1]
    bb = Box(200, 300, 800, 1290)
    left = bb.x0 - p.margin.crop.x0
    right = p.margin.crop.x1 - bb.x1
    assert abs(left - right) < 2.0


def test_body_text_box_excludes_nombre_overlap():
    # The nombre is never body text: a TEXT region overlapping the detected
    # nombre box must not widen the body hull (it did on 111/128 real pages).
    from hokusai_press.geometry.margin import body_text_box
    from hokusai_press.model import Region, RegionKind

    p = _page(0, Box(200, 300, 820, 1330), nombre=Box(770, 1300, 820, 1330))
    p.regions = [
        Region(kind=RegionKind.TEXT, box=Box(200, 300, 700, 1290)),  # body
        Region(kind=RegionKind.TEXT, box=Box(768, 1298, 822, 1332)),  # nombre
    ]
    bb = body_text_box(p)
    assert bb.x1 <= 700


def test_body_text_box_tategaki_strips_bottom_furniture_via_y_spans():
    # Tategaki columns: no dominant x-span, but the columns share the full
    # body height -> the y-span pass must strip a short bottom furniture
    # block (running head) that x-span analysis alone could not.
    from hokusai_press.geometry.margin import body_text_box
    from hokusai_press.model import Region, RegionKind

    cols = [Box(200 + i * 90, 300, 260 + i * 90, 1200) for i in range(5)]
    furniture = Box(240, 1290, 420, 1330)     # short, separated, bottom
    p = _page(0, Box(200, 300, 710, 1330))
    p.regions = [Region(kind=RegionKind.TEXT, box=b)
                 for b in cols + [furniture]]
    bb = body_text_box(p)
    assert bb.y1 <= 1200


def test_running_head_overlapping_body_is_stripped():
    # Real-corpus bug (img20260416 p7/p8): a 柱 at the top fore corner
    # OVERLAPS the body horizontally, so the x-pass alone can never strip it;
    # the hull then extends toward the fore edge and centering it mirrors the
    # body ~3mm off-center on every verso/recto pair. The y-pass (outer short
    # spans) must remove it even though it x-overlaps the body.
    from hokusai_press.geometry.margin import body_text_box

    body_lines = [Box(200, 300 + i * 50, 800, 340 + i * 50) for i in range(20)]
    kashira = Box(600, 100, 950, 140)      # top fore corner, x-overlaps body
    p = _page(0, Box(200, 100, 950, 1340))
    _with_regions(p, body_lines + [kashira])

    bb = body_text_box(p)
    assert bb.x1 <= 800          # 柱's fore-edge protrusion excluded
    assert bb.y0 >= 300          # and its top band too


def test_interior_heading_is_kept():
    # Outer-only stripping: a short heading BETWEEN body blocks is body text,
    # not furniture -- it must stay in the hull.
    from hokusai_press.geometry.margin import body_text_box

    upper = [Box(200, 300 + i * 50, 800, 340 + i * 50) for i in range(8)]
    heading = Box(200, 760, 500, 800)      # short, separated, interior
    lower = [Box(200, 860 + i * 50, 800, 900 + i * 50) for i in range(8)]
    p = _page(0, Box(200, 300, 800, 1300))
    _with_regions(p, upper + [heading] + lower)

    bb = body_text_box(p)
    assert bb.y0 <= 300 and bb.y1 >= 1250   # last lower line ends at 1250
