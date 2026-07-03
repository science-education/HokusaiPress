"""rotate_page_params: exact box permutation under 90-deg-step rotation."""

from hokusai_press.model import (
    Box,
    Deskew,
    Margin,
    PageParams,
    Region,
    RegionKind,
    SourceRef,
    rotate_page_params,
)


def _page():
    return PageParams(
        source=SourceRef(path="s.pdf", page_index=0),
        deskew=Deskew(angle_deg=0.4, confidence=2.0),
        margin=Margin(content=Box(10, 20, 110, 220), page_w=300, page_h=400,
                      nombre_box=Box(140, 380, 160, 395)),
        regions=[Region(kind=RegionKind.TEXT, box=Box(10, 20, 110, 40))],
    )


def test_rotate_90_cw_maps_corners_exactly():
    p = _page()
    rotate_page_params(p, 90, 300, 400)   # W=300, H=400 -> new frame 400x300
    b = p.regions[0].box
    # (x,y) -> (H - y, x): (10,20)->(380,10), (110,40)->(360,110)
    assert (b.x0, b.y0, b.x1, b.y1) == (360, 10, 380, 110)
    assert p.rotation == 90
    assert (p.margin.page_w, p.margin.page_h) == (400, 300)
    assert p.deskew.angle_deg == 0.0  # measured against old axes


def test_rotate_180_keeps_deskew_and_dims():
    p = _page()
    rotate_page_params(p, 180, 300, 400)
    b = p.regions[0].box
    assert (b.x0, b.y0, b.x1, b.y1) == (190, 360, 290, 380)
    assert p.deskew.angle_deg == 0.4
    assert (p.margin.page_w, p.margin.page_h) == (300, 400)


def test_four_quarter_turns_round_trip():
    p = _page()
    w, h = 300, 400
    for _ in range(4):
        rotate_page_params(p, 90, w, h)
        w, h = h, w
    b = p.regions[0].box
    assert (b.x0, b.y0, b.x1, b.y1) == (10, 20, 110, 40)
    assert p.rotation == 0
    nb = p.margin.nombre_box
    assert (nb.x0, nb.y0, nb.x1, nb.y1) == (140, 380, 160, 395)
