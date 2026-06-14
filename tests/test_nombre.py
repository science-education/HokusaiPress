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


# ---- parsing -------------------------------------------------------------

def test_parse_arabic_halfwidth_and_fullwidth():
    assert nombre.parse_numeral("12") == 12
    assert nombre.parse_numeral("１２３") == 123
    assert nombre.parse_numeral(" 7 ") == 7


def test_parse_kanji_positional_and_structured():
    assert nombre.parse_numeral("一二三") == 123
    assert nombre.parse_numeral("十二") == 12
    assert nombre.parse_numeral("二百五") == 205
    assert nombre.parse_numeral("〇") == 0


def test_parse_roman():
    assert nombre.parse_roman("i") == 1
    assert nombre.parse_roman("iv") == 4
    assert nombre.parse_roman("viii") == 8
    assert nombre.parse_roman("xii") == 12
    assert nombre.parse_roman("Xl") == 40       # case-insensitive
    assert nombre.parse_roman("hello") is None


def test_parse_rejects_non_numbers():
    assert nombre.parse_numeral("はじめに") is None
    assert nombre.parse_numeral("") is None
    assert nombre.parse_numeral("1234567") is None     # too long for a nombre


# ---- resolution helpers --------------------------------------------------

def _num_box(h):                  # a box centered in the bottom band
    return Box(480, h * 0.95, 520, h * 0.95 + 24)


def _page(idx, *, text=None, h=1000, body=True, extra=None):
    regions = []
    if body:
        regions.append(Region(kind=RegionKind.TEXT, box=Box(100, 400, 900, 430),
                              ocr_text="本文のテキスト行", source="dbnet"))
    if text is not None:
        regions.append(Region(kind=RegionKind.TEXT, box=_num_box(h),
                               ocr_text=text, source="dbnet"))
    if extra is not None:         # an extra (band) token, e.g. a running header
        box, t = extra
        regions.append(Region(kind=RegionKind.TEXT, box=box, ocr_text=t,
                               source="dbnet"))
    return PageParams(
        source=SourceRef(path="x.pdf", page_index=idx),
        margin=Margin(content=Box(80, 80, 920, 920), confidence=0.8),
        regions=regions,
    )


# ---- robust resolution ---------------------------------------------------

def test_resolve_arabic_run_sets_numbers():
    # a clean body run 1..10 (offset 0 from index)
    pages = [_page(i, text=str(i + 1)) for i in range(10)]
    warnings = nombre.resolve(pages, [1000] * 10)
    assert [p.page_number for p in pages] == list(range(1, 11))
    assert warnings == []
    assert all(p.margin.nombre_box is not None for p in pages)


def test_resolve_unnumbered_front_matter_stays_none():
    # 3 unnumbered front-matter pages, then body 1..8
    pages = [_page(i, text=None) for i in range(3)]
    pages += [_page(3 + i, text=str(i + 1)) for i in range(8)]
    nombre.resolve(pages, [1000] * len(pages))
    assert [p.page_number for p in pages[:3]] == [None, None, None]
    assert [p.page_number for p in pages[3:]] == list(range(1, 9))


def test_resolve_roman_front_matter_then_arabic_body():
    roman = ["i", "ii", "iii", "iv", "v"]
    pages = [_page(i, text=roman[i]) for i in range(5)]
    pages += [_page(5 + i, text=str(i + 1)) for i in range(8)]
    nombre.resolve(pages, [1000] * len(pages))
    assert [p.page_number for p in pages[:5]] == [1, 2, 3, 4, 5]   # roman run
    assert [p.page_number for p in pages[5:]] == list(range(1, 9))  # arabic run


def test_resolve_rejects_running_header_and_stray_zero():
    # body run 1..10, but every page also has a running header "11" (a chapter
    # number), and one page has a stray "0". Neither must be taken as the nombre.
    pages = []
    for i in range(10):
        header = (Box(60, 30, 100, 54), "11")   # constant header, top band
        pages.append(_page(i, text=str(i + 1), extra=header))
    pages[4].regions.append(Region(kind=RegionKind.TEXT, box=_num_box(1000),
                                   ocr_text="0", source="dbnet"))
    warnings = nombre.resolve(pages, [1000] * 10)
    assert [p.page_number for p in pages] == list(range(1, 11))  # clean
    assert warnings == []                                        # no false gap


def test_resolve_warns_on_real_missing_pages():
    # body numbered 1..6 then jumps to 10..13 with contiguous indices => pages
    # 7,8,9 are physically missing (an offset shift between two supported runs).
    nums = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
    pages = [_page(i, text=str(nums[i])) for i in range(len(nums))]
    warnings = nombre.resolve(pages, [1000] * len(nums))
    assert any("missing" in w for w in warnings)
    assert any(Flag.PAGE_NUMBER_GAP in p.flags for p in pages) or warnings


def test_resolve_no_numbers_leaves_pages_untouched():
    pages = [_page(i, text=None) for i in range(4)]
    assert nombre.resolve(pages, [1000] * 4) == []
    assert all(p.page_number is None for p in pages)
