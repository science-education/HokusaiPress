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


def _box_at(xc, yc, w=1000, h=1000):
    return Box(xc - 15, yc - 10, xc + 15, yc + 10)


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
    warnings = nombre.resolve(pages, [1000] * 10, [1000] * 10)
    assert [p.page_number for p in pages] == list(range(1, 11))
    assert warnings == []
    assert all(p.margin.nombre_box is not None for p in pages)


def test_resolve_unnumbered_front_matter_stays_none():
    # 3 unnumbered front-matter pages, then body 1..8
    pages = [_page(i, text=None) for i in range(3)]
    pages += [_page(3 + i, text=str(i + 1)) for i in range(8)]
    nombre.resolve(pages, [1000] * len(pages), [1000] * len(pages))
    assert [p.page_number for p in pages[:3]] == [None, None, None]
    assert [p.page_number for p in pages[3:]] == list(range(1, 9))


def test_fused_header_digits_recovered():
    from hokusai_press.model import Box, Region, RegionKind
    from hokusai_press.nombre import _fused_digit_candidates

    # page number fused with the running header into one region
    r = Region(kind=RegionKind.TEXT, box=Box(100, 80, 700, 130),
               ocr_text="112第3部")
    cands = _fused_digit_candidates(r, 1000)
    assert 112 in [v for v, _ in cands]
    sub = next(b for v, b in cands if v == 112)
    assert sub.x0 == 100 and sub.x1 < 700      # sub-box hugs the left edge
    # a region with no edge digits yields nothing
    r2 = Region(kind=RegionKind.TEXT, box=Box(0, 0, 100, 20), ocr_text="第6章")
    assert _fused_digit_candidates(r2, 1000) == []


def test_resolve_roman_front_matter_then_arabic_body():
    roman = ["i", "ii", "iii", "iv", "v"]
    pages = [_page(i, text=roman[i]) for i in range(5)]
    pages += [_page(5 + i, text=str(i + 1)) for i in range(8)]
    nombre.resolve(pages, [1000] * len(pages), [1000] * len(pages))
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
    warnings = nombre.resolve(pages, [1000] * 10, [1000] * 10)
    assert [p.page_number for p in pages] == list(range(1, 11))  # clean
    assert warnings == []                                        # no false gap


def test_resolve_warns_on_real_missing_pages():
    # body numbered 1..6 then jumps to 10..13 with contiguous indices => pages
    # 7,8,9 are physically missing (an offset shift between two supported runs).
    nums = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
    pages = [_page(i, text=str(nums[i])) for i in range(len(nums))]
    warnings = nombre.resolve(pages, [1000] * len(nums), [1000] * len(nums))
    assert any("missing" in w for w in warnings)
    assert any(Flag.PAGE_NUMBER_GAP in p.flags for p in pages) or warnings


def test_resolve_clears_stale_nombre_on_unread_pages():
    # body run 1..10, but one page's number is unread; its (wrong) geometric
    # nombre box must be cleared, not left pointing at a stray spot.
    pages = [_page(i, text=str(i + 1)) for i in range(10)]
    pages[4].regions = [r for r in pages[4].regions if r.ocr_text != "5"]  # unread
    pages[4].margin.nombre_box = Box(900, 50, 950, 80)   # stale geometric guess
    nombre.resolve(pages, [1000] * 10, [1000] * 10)
    assert pages[4].page_number is None
    assert pages[4].margin.nombre_box is None            # stale box cleared


def test_resolve_no_numbers_leaves_pages_untouched():
    pages = [_page(i, text=None) for i in range(4)]
    assert nombre.resolve(pages, [1000] * 4, [1000] * 4) == []
    assert all(p.page_number is None for p in pages)


def test_resolve_position_model_accepts_alternating_corners():
    pages = []
    for i in range(20):
        xc = 850 if i % 2 == 0 else 80
        extra = (Box(470, 945, 500, 965), "40")  # centered distractor
        p = _page(i, text=None, extra=extra)
        p.regions.append(Region(kind=RegionKind.TEXT,
                                box=_box_at(xc, 950),
                                ocr_text=str(i + 1), source="dbnet"))
        pages.append(p)

    warnings = nombre.resolve(pages, [1000] * 20, [1000] * 20)

    assert warnings == []
    assert [p.page_number for p in pages] == list(range(1, 21))
    assert pages[0].margin.nombre_box.x0 > 800
    assert pages[1].margin.nombre_box.x1 < 120


def test_resolve_position_model_accepts_centered_layout():
    pages = []
    for i in range(20):
        p = _page(i, text=None)
        p.regions.append(Region(kind=RegionKind.TEXT,
                                box=_box_at(500, 950),
                                ocr_text=str(i + 1), source="dbnet"))
        pages.append(p)

    nombre.resolve(pages, [1000] * 20, [1000] * 20)

    assert [p.page_number for p in pages] == list(range(1, 21))
    assert all(480 <= p.margin.nombre_box.x0 <= 520 for p in pages)


def test_resolve_position_model_rejects_off_position_table_cluster():
    pages = []
    for i in range(20):
        p = _page(i, text=None)
        p.regions.append(Region(kind=RegionKind.TEXT,
                                box=_box_at(500, 950),
                                ocr_text=str(i + 1), source="dbnet"))
        p.regions.append(Region(kind=RegionKind.TEXT,
                                box=_box_at(160, 920),
                                ocr_text=str(i + 1), source="dbnet"))
        pages.append(p)

    nombre.resolve(pages, [1000] * 20, [1000] * 20)

    assert [p.page_number for p in pages] == list(range(1, 21))
    assert all(480 <= p.margin.nombre_box.x0 <= 520 for p in pages)


def test_resolve_position_model_corrects_ocr_misread_to_expected_number():
    pages = []
    for i in range(20):
        text = "14" if i == 9 else str(i + 1)
        p = _page(i, text=None)
        p.regions.append(Region(kind=RegionKind.TEXT,
                                box=_box_at(500, 950),
                                ocr_text=text, source="dbnet"))
        pages.append(p)

    warnings = nombre.resolve(pages, [1000] * 20, [1000] * 20)

    assert pages[9].page_number == 10
    assert pages[9].nombre_text == "14"
    assert warnings == []


def test_resolve_position_model_preserves_genuine_gap_warning():
    nums = list(range(1, 11)) + list(range(14, 24))
    pages = []
    for i, n in enumerate(nums):
        p = _page(i, text=None)
        p.regions.append(Region(kind=RegionKind.TEXT,
                                box=_box_at(500, 950),
                                ocr_text=str(n), source="dbnet"))
        pages.append(p)

    warnings = nombre.resolve(pages, [1000] * 20, [1000] * 20)

    assert any("missing" in w for w in warnings)
    assert any(Flag.PAGE_NUMBER_GAP in p.flags for p in pages)


def test_resolve_vertical_folios_uses_reversed_digits_and_sequence_repair():
    # Real Japanese vertical-book pattern: OCR may reverse stacked digits
    # (12 -> 21, 13 -> 31), then partially read later numbers (16 -> 10,
    # 17 -> "7."). Stable alternating-corner geometry must recover 11..17.
    texts = ["11", "21", "31", "14", "15", "10", "7."]
    pages = []
    for i, text in enumerate(texts):
        x = 100 if i % 2 == 0 else 900
        p = _page(i, text=None)
        p.regions.append(Region(
            kind=RegionKind.TEXT, box=_box_at(x, 45),
            ocr_text=text, ocr_conf=0.9, source="ndlocr"))
        pages.append(p)

    nombre.resolve(pages, [1000] * len(pages), [1000] * len(pages))

    assert [p.page_number for p in pages] == list(range(11, 18))


def test_resolve_fused_roman_running_head_before_arabic_body():
    pages = [_page(0, text=None), _page(1, text=None)]
    pages[0].regions.append(Region(
        kind=RegionKind.TEXT, box=_box_at(100, 45, w=1000),
        ocr_text="iii 目次", ocr_conf=0.9, source="ndlocr"))
    pages[1].regions.append(Region(
        kind=RegionKind.TEXT, box=_box_at(900, 45, w=1000),
        ocr_text="iv", ocr_conf=0.9, source="ndlocr"))
    for i, value in enumerate(range(11, 15), start=2):
        x = 100 if i % 2 == 0 else 900
        p = _page(i, text=None)
        p.regions.append(Region(
            kind=RegionKind.TEXT, box=_box_at(x, 45),
            ocr_text=str(value), ocr_conf=0.9, source="ndlocr"))
        pages.append(p)

    nombre.resolve(pages, [1000] * len(pages), [1000] * len(pages))

    assert [p.page_number for p in pages] == [3, 4, 11, 12, 13, 14]
    assert pages[0].nombre_text == "iii 目次"


def test_reader_series_overrules_spurious_leading_digit_and_fills_roman_hole():
    pages = [_page(i, text=None) for i in range(11)]
    # Main OCR overreads a leading stroke/digit on vertical Arabic folios.
    for i, text in enumerate(["11", "21", "31", "14", "15", "10", "17"], 4):
        x = 100 if i % 2 == 0 else 900
        pages[i].regions.append(Region(
            kind=RegionKind.TEXT, box=_box_at(x, 45), ocr_text=text,
            ocr_conf=0.9, source="ndlocr-folio"))
    # The independent compact reader sees the stable units sequence correctly.
    for i, value in enumerate(range(1, 8), 4):
        x = 100 if i % 2 == 0 else 900
        region = Region(kind=RegionKind.TEXT, box=_box_at(x, 45),
                        ocr_text=str(value), ocr_conf=0.99,
                        source="nombre_reader")
        pages[i]._nombre_candidates = [("top", "num", value, region)]
    # i/iii/iv are visible, while ii is unreadable but bracketed.
    for i, text, value in ((0, "i", 1), (2, "iii", 3), (3, "iv", 4)):
        x = 100 if i % 2 == 0 else 900
        region = Region(kind=RegionKind.TEXT, box=_box_at(x, 45),
                        ocr_text=text, ocr_conf=0.99,
                        source="nombre_reader")
        pages[i]._nombre_candidates = [("top", "roman", value, region)]

    nombre.resolve(pages, [1000] * len(pages), [1000] * len(pages))

    assert [p.page_number for p in pages] == [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7]
    assert [p.nombre_text for p in pages[:4]] == ["i", "ii", "iii", "iv"]
    # Logical labels may be inferred on front matter, but a weak fallback probe
    # must not fabricate a printed folio box on the cover.
    assert pages[0].margin.nombre_box is None


def test_pre_toc_prior_does_not_erase_strong_observed_bottom_folios():
    pages = [_page(i, text=None) for i in range(7)]
    for i, value in enumerate(range(1, 6), start=1):
        x = 100 if i % 2 == 0 else 900
        region = Region(kind=RegionKind.TEXT, box=_box_at(x, 950),
                        ocr_text=str(value), ocr_conf=0.99,
                        source="nombre_reader")
        pages[i]._nombre_candidates = [("bottom", "num", value, region)]
    pages[6].regions.append(Region(
        kind=RegionKind.TEXT, box=Box(100, 100, 300, 130),
        ocr_text="目次", source="ndlocr"))

    nombre.resolve(pages, [1000] * len(pages), [1000] * len(pages))

    assert [p.page_number for p in pages[1:6]] == [1, 2, 3, 4, 5]
    assert all(p.margin.nombre_box is not None for p in pages[1:6])


def test_finalize_pagination_interpolates_only_bracketed_holes_and_keeps_restart():
    pages = [_page(i, text=None) for i in range(5)]
    for page, predicted in zip(pages, [6, 7, 8, None, 10]):
        page.page_number = predicted

    def evidence(page, value, *, channel="main_ocr"):
        box = _box_at(100 if page.source.page_index % 2 == 0 else 900, 45)
        page.margin.nombre_box = box
        page.nombre_evidence[channel] = [{
            "text": str(value), "confidence": 0.95,
            "box": {"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1},
            "values": [{"kind": "num", "value": value, "variant": "direct"}],
        }]

    evidence(pages[0], 6)
    evidence(pages[2], 8)
    evidence(pages[4], 26, channel="main_ocr")
    evidence(pages[4], 26, channel="dedicated_ocr")

    nombre.finalize_pagination(pages)

    assert [p.page_number for p in pages] == [6, 7, 8, None, 26]
    assert pages[1].pagination.role.value == "counted_unprinted"
    assert pages[1].pagination.supporting_pages == [1, 3]
    assert pages[3].pagination.role.value == "uncounted"
    assert pages[3].pagination.pdf_label == "scan-4"
    assert pages[4].pagination.conflict is True
    assert pages[4].pagination.predicted_number == 10
    assert pages[4].nombre_text == "26"
