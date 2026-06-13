from hokusai_press import nombre
from hokusai_press.model import (
    Box,
    Flag,
    Margin,
    PageParams,
    Region,
    RegionKind,
    SourceRef,
)


def test_parse_arabic_halfwidth_and_fullwidth():
    assert nombre.parse_numeral("12") == 12
    assert nombre.parse_numeral("１２３") == 123
    assert nombre.parse_numeral(" 7 ") == 7


def test_parse_kanji_positional_and_structured():
    assert nombre.parse_numeral("一二三") == 123      # positional
    assert nombre.parse_numeral("十二") == 12          # structured
    assert nombre.parse_numeral("二百五") == 205
    assert nombre.parse_numeral("〇") == 0


def test_parse_rejects_non_numbers():
    assert nombre.parse_numeral("はじめに") is None
    assert nombre.parse_numeral("") is None
    assert nombre.parse_numeral("1234567") is None     # too long for a nombre


def _page(idx, h=1000, *, num_box=None, num_text=None, body=True):
    regions = []
    if body:
        regions.append(Region(kind=RegionKind.TEXT, box=Box(100, 400, 900, 430),
                              ocr_text="本文のテキスト行", source="dbnet"))
    if num_box is not None:
        regions.append(Region(kind=RegionKind.TEXT, box=num_box,
                               ocr_text=num_text, source="dbnet"))
    return PageParams(
        source=SourceRef(path="x.pdf", page_index=idx),
        margin=Margin(content=Box(80, 80, 920, 920), confidence=0.8),
        regions=regions,
    )


def test_resolve_picks_bottom_band_and_sets_page_number():
    # page numbers 1,2,3 in the bottom band (y near 960 on a 1000-tall page)
    pages = [_page(i, num_box=Box(480, 950, 520, 980), num_text=str(i + 1))
             for i in range(3)]
    warnings = nombre.resolve(pages, [1000, 1000, 1000])
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert warnings == []
    # nombre box snapped to the number's region
    assert all(p.margin.nombre_box is not None for p in pages)


def test_resolve_ignores_body_and_out_of_band_numbers():
    # a stray "2024" in the body must not be taken as the nombre
    pages = [_page(0, num_box=Box(480, 950, 520, 980), num_text="5")]
    pages[0].regions.append(Region(kind=RegionKind.TEXT,
                                   box=Box(300, 500, 420, 530),
                                   ocr_text="2024", source="dbnet"))
    nombre.resolve(pages, [1000])
    assert pages[0].page_number == 5


def test_resolve_warns_on_sequence_gap():
    # 10 -> 13 with no unnumbered pages between => likely missing pages
    pages = [
        _page(0, num_box=Box(480, 950, 520, 980), num_text="10"),
        _page(1, num_box=Box(480, 950, 520, 980), num_text="13"),
        _page(2, num_box=Box(480, 950, 520, 980), num_text="14"),
    ]
    warnings = nombre.resolve(pages, [1000, 1000, 1000])
    assert any("missing" in w for w in warnings)
    assert Flag.PAGE_NUMBER_GAP in pages[1].flags


def test_resolve_no_numbers_leaves_pages_untouched():
    pages = [_page(0), _page(1)]            # no numeric tokens anywhere
    warnings = nombre.resolve(pages, [1000, 1000])
    assert warnings == []
    assert all(p.page_number is None for p in pages)
