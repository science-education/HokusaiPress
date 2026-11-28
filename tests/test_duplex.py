"""Unit tests for duplex sheet-pair alignment, skew, and blank-page insertion."""

from __future__ import annotations

import numpy as np
import pypdfium2 as pdfium
import pytest

from hokusai_press.duplex import (
    align_after_blank_insertions,
    align_illustrated_folios,
    apply_duplex_skew_exceptions,
    apply_duplex_translation_exceptions,
    detect_blank_insertions,
    determine_duplex_pairing,
    logical_page_indices,
)
from hokusai_press.model import (
    Box,
    Deskew,
    Document,
    Flag,
    Margin,
    PageParams,
    Region,
    RegionKind,
    SourceRef,
)
from hokusai_press.render import build_pdf


def _make_duplex_book(n_pages: int = 30, parity: int = 0, weak_corr: bool = False):
    """Build synthetic book with realistic anti-correlated X shifts on sheets."""
    rng = np.random.default_rng(42)
    pages: list[PageParams] = []
    pw, ph = 2000, 3000
    cw, ch = 1600.0, 2400.0
    dpi = 300.0
    px_per_mm = dpi / 25.4
    # Sheet-level shifts in mm
    sheet_dx = rng.normal(0.0, 1.5, size=n_pages)
    sheet_dy = rng.normal(0.0, 0.8, size=n_pages)

    for i in range(n_pages):
        sheet_id = (i - parity) // 2
        is_front = (i % 2 == parity)
        dx_mm = sheet_dx[sheet_id] if is_front else -sheet_dx[sheet_id]
        dy_mm = sheet_dy[sheet_id]
        if weak_corr:
            dx_mm = float(rng.normal(0.0, 1.5))
        cx = pw / 2.0 + dx_mm * px_per_mm
        cy = ph / 2.0 + dy_mm * px_per_mm
        crop = Box(cx - cw / 2.0, cy - ch / 2.0, cx + cw / 2.0, cy + ch / 2.0)
        # Body folio: parity=0 -> odd folio at even index; parity=1 -> odd folio at odd index
        folio = i + 1 if parity == 0 else i
        p = PageParams(
            source=SourceRef(path="dummy.pdf", page_index=i),
            dpi=dpi,
            deskew=Deskew(angle_deg=0.0, confidence=5.0),
            margin=Margin(
                content=Box(250, 350, 1750, 2650),
                confidence=0.9,
                nombre_box=Box(1000, 2800, 1040, 2830),
                nombre_source="folio",
                crop=crop,
                page_w=pw,
                page_h=ph,
                target_w=round(cw),
                target_h=round(ch),
            ),
            regions=[
                Region(
                    kind=RegionKind.TEXT,
                    box=Box(400, 500, 1600, 2500),
                    ocr_text="本文テキスト",
                )
            ],
            page_number=folio if folio >= 1 else None,
            page_number_source="folio" if folio >= 1 else None,
        )
        pages.append(p)
    return pages


def test_duplex_translation_partner_median_uncorrected():
    pages = _make_duplex_book(n_pages=30, parity=0)
    # Case 1: Page 4 (partner=5) loses folio and is displaced by -15mm -> partner copy fixes it
    p4 = pages[4]
    p4.page_number = None
    p4.page_number_source = None
    p4.margin.nombre_box = None
    p4.margin.nombre_source = None
    px_per_mm = 300.0 / 25.4
    p4.margin.crop = Box(
        p4.margin.crop.x0 - 15.0 * px_per_mm,
        p4.margin.crop.y0 - 10.0 * px_per_mm,
        p4.margin.crop.x1 - 15.0 * px_per_mm,
        p4.margin.crop.y1 - 10.0 * px_per_mm,
    )
    orig_p5_crop = pages[5].margin.crop

    # Case 2: Page 6 (partner=7) has partner with large +5mm shift so partner-copy clips text,
    # while median (0mm) does NOT clip -> falls back to median.
    p6 = pages[6]
    p6.page_number = None
    p6.page_number_source = None
    p6.margin.nombre_box = None
    p6.margin.nombre_source = None
    # Partner p7 shifted right by +8mm -> p6 partner-copy shifts left by -8mm
    p7 = pages[7]
    p7.margin.crop = Box(
        p7.margin.crop.x0 + 8.0 * px_per_mm, p7.margin.crop.y0,
        p7.margin.crop.x1 + 8.0 * px_per_mm, p7.margin.crop.y1,
    )
    # Place p6's OCR text just inside median crop's right edge (x1=1800) so -8mm clips it,
    # but median crop (x1=1800) does NOT clip it.
    p6.regions = [Region(kind=RegionKind.TEXT, box=Box(1750, 1000, 1795, 1100), ocr_text="端の文字")]

    # Case 3: Page 8 (partner=9) has OCR text outside both partner-copy and median -> uncorrected
    p8 = pages[8]
    p8.page_number = None
    p8.page_number_source = None
    p8.margin.nombre_box = None
    p8.margin.nombre_source = None
    p8.regions = [Region(kind=RegionKind.TEXT, box=Box(1850, 1000, 1900, 1100), ocr_text="さらに外側の文字")]
    p8_base = Box(p8.margin.crop.x0, p8.margin.crop.y0, p8.margin.crop.x1, p8.margin.crop.y1)

    pairing = determine_duplex_pairing(pages)
    assert pairing.known and pairing.parity == 0
    res = apply_duplex_translation_exceptions(pages, pairing)

    assert 4 in res["partner"]
    assert 6 in res["median"]
    assert 8 in res["uncorrected"]
    # Partner page 5 is untouched
    assert pages[5].margin.crop == orig_p5_crop
    # Uncorrected page 8 keeps base crop and gets warning flag
    assert pages[8].margin.crop == p8_base
    assert Flag.MARGIN_INCONSISTENT in pages[8].flags


@pytest.mark.parametrize("kind", [RegionKind.PHOTO, RegionKind.FIGURE])
@pytest.mark.parametrize("case", ["align", "clip", "blank", "known_box", "broken_run", "caption"])
def test_align_illustrated_folio_from_local_numbered_run(kind, case):
    pages = _make_duplex_book()
    for p in pages:
        p.page_number += 20
        m = p.margin
        m.crop = Box(150, 200, 1750, 2900)
        m.target_h = m.crop.height
        m.nombre_box = (
            Box(320, 2800, 334, 2830) if p.page_number % 2 else Box(1560, 2800, 1574, 2830)
        )
    idx = 8
    page = pages[idx]
    m = page.margin
    plate = Box(280, 500, 1670, 2550)
    m.content = Box(280, 500, 1670, 2830)
    page.regions = [Region(kind, plate)]
    if case != "known_box":
        m.nombre_box = None
        m.nombre_source = None
    if case == "broken_run":
        pages[idx + 1].page_number += 1
    if case == "clip":
        # Crop must keep the entire plate even when there is no OCR text.
        m.content.x0 = 152
        plate.x0 = 152
    if case == "caption":
        page.regions.append(Region(RegionKind.TEXT, Box(325, 2770, 600, 2850), ocr_text="caption"))
    image = np.full((3000, 2000), 255, dtype=np.uint8)
    if case != "blank":
        image[2800:2830, 330:339] = 0
        image[2800:2830, 346:355] = 0
    crops = [p.margin.crop for p in pages]
    loaded = []

    def load(index):
        loaded.append(index)
        return image

    changed = align_illustrated_folios(pages, load)
    assert loaded == ([] if case in ("known_box", "broken_run") else [idx])
    assert page.page_number == 29
    if case == "align":
        assert changed == [idx]
        assert m.crop == Box(160, 200, 1760, 2900)
        assert m.nombre_box is None
        assert align_illustrated_folios(pages, load) == []
        crops[idx] = m.crop
    else:
        assert changed == []
    assert [p.margin.crop for p in pages] == crops


def test_unknown_pairing_leaves_pages_untouched():
    pages = _make_duplex_book(n_pages=30, parity=0, weak_corr=True)
    p4 = pages[4]
    p4.page_number = None
    p4.page_number_source = None
    p4.margin.crop = Box(50, 50, 1650, 2450)
    before = Box(50, 50, 1650, 2450)

    pairing = determine_duplex_pairing(pages)
    assert not pairing.known
    assert pairing.status == "不明"

    res = apply_duplex_translation_exceptions(pages, pairing)
    assert not res["applied"]
    assert pages[4].margin.crop == before


def test_skew_pair_scoring_ignores_scanner_streak():
    import cv2

    pages = _make_duplex_book(n_pages=24, parity=0)
    pairing = determine_duplex_pairing(pages)
    assert pairing.known

    # Create synthetic front (tilted so deskew -5.0 restores horizontal)
    # and back (tilted opposite so deskew +5.0 restores horizontal, PLUS vertical streak!)
    h, w = 1000, 700
    base = np.full((h, w, 3), 255, dtype=np.uint8)
    for y in range(150, 850, 40):
        cv2.line(base, (100, y), (600, y), (0, 0, 0), 3)

    M_front = cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 1.0)
    img_front = cv2.warpAffine(base, M_front, (w, h), borderValue=(255, 255, 255))

    M_back = cv2.getRotationMatrix2D((w / 2, h / 2), -5.0, 1.0)
    img_back = cv2.warpAffine(base, M_back, (w, h), borderValue=(255, 255, 255))
    # Draw full-length vertical scanner streak on back image
    img_back[:, 348:354] = 140

    pages[0].deskew = Deskew(angle_deg=-5.0, confidence=4.0)
    pages[1].deskew = Deskew(angle_deg=0.0, confidence=9.5)  # fooled by streak!

    imgs = {0: img_front, 1: img_back}
    changes = apply_duplex_skew_exceptions(pages, pairing, lambda idx: imgs[idx])
    assert len(changes) == 1
    assert changes[0]["page_index"] == 1
    assert abs(pages[1].deskew.angle_deg - 5.0) < 0.01
    assert changes[0]["res_diff"] >= 0.3
    assert abs(changes[0]["res_cand"]) <= 0.6


def test_skew_does_not_corrupt_good_page_when_partner_is_wrong():
    import cv2

    pages = _make_duplex_book(n_pages=24, parity=0)
    pairing = determine_duplex_pairing(pages)
    assert pairing.known

    h, w = 1000, 700
    # Page 0: genuinely straight horizontal text (angle_deg = 0.0)
    img_good = np.full((h, w, 3), 255, dtype=np.uint8)
    for y in range(150, 850, 40):
        cv2.line(img_good, (100, y), (600, y), (0, 0, 0), 3)

    # Page 1: genuinely straight text too, but mis-estimated at +1.5 deg
    # (e.g., sparse text / figure fooled find_skew). Its sharpness at 0.0 is
    # higher than at +1.5, OR if it were truly tilted +1.5 deg, page 0's own
    # score at -1.5 deg is much lower than at 0.0 deg -> page 0 must NOT change!
    M_tilt = cv2.getRotationMatrix2D((w / 2, h / 2), -1.5, 1.0)
    img_tilted = cv2.warpAffine(img_good, M_tilt, (w, h), borderValue=(255, 255, 255))

    pages[0].deskew = Deskew(angle_deg=0.0, confidence=8.0)
    pages[1].deskew = Deskew(angle_deg=1.5, confidence=1.0)

    imgs = {0: img_good, 1: img_tilted}
    changes = apply_duplex_skew_exceptions(pages, pairing, lambda idx: imgs[idx])
    # Page 0 stays at 0.0 deg (never tilted to -1.5 deg)
    assert pages[0].deskew.angle_deg == 0.0
    assert all(c["page_index"] != 0 for c in changes)


def test_skew_unmeasurable_sparse_page_is_not_changed():
    pages = _make_duplex_book(n_pages=24, parity=0)
    pairing = determine_duplex_pairing(pages)
    assert pairing.known

    h, w = 1000, 700
    img_sparse = np.full((h, w, 3), 255, dtype=np.uint8)
    # Tiny 3-character mark: too sparse to measure skew reliably
    img_sparse[100:130, 100:115] = 0
    img_back = np.full((h, w, 3), 255, dtype=np.uint8)

    pages[0].deskew = Deskew(angle_deg=2.5, confidence=1.0)
    pages[1].deskew = Deskew(angle_deg=-0.1, confidence=5.0)

    imgs = {0: img_sparse, 1: img_back}
    changes = apply_duplex_skew_exceptions(pages, pairing, lambda idx: imgs[idx])
    assert pages[0].deskew.angle_deg == 2.5
    assert changes == []


def test_blank_insertion_preserves_dimensions_and_text_layer(tmp_path):
    pages = _make_duplex_book(n_pages=6, parity=1)
    # Shifted book: page 0 is title (no folio), page 1 has folio=1 (first_fn=1)
    pages[0].page_number = None
    pages[0].page_number_source = None
    pairing = DuplexPairing_or_mock = determine_duplex_pairing(_make_duplex_book(n_pages=24, parity=1))
    assert pairing.known and pairing.parity == 1

    ins = detect_blank_insertions(pages[:4], pairing)
    assert ins["insert_before"] == [1]

    # Build 4-page PDF with 1 blank inserted before index 1 -> 5 pages total
    originals = [np.full((3000, 2000, 3), 255, dtype=np.uint8) for _ in range(4)]
    # Put a distinguishable dark mark on page 1's region
    originals[1][600:650, 600:650] = 0
    doc = Document(source_path="dummy.pdf", pages=pages[:4], blank_insert_before=[1])
    doc.render.target_dpi = 300
    out_pdf = str(tmp_path / "out_blank.pdf")
    build_pdf(doc, originals, out_pdf)

    pdf = pdfium.PdfDocument(out_pdf)
    try:
        assert len(pdf) == 5
        sizes = [pdf[k].get_size() for k in range(5)]
        assert all(abs(s[0] - sizes[0][0]) < 0.1 and abs(s[1] - sizes[0][1]) < 0.1 for s in sizes)
        # Inserted blank at output index 1 has no text
        assert pdf[1].get_textpage().get_text_range().strip() == ""
        # Original page 1 (now at output index 2) keeps its OCR text
        assert "本文テキスト" in "".join(pdf[2].get_textpage().get_text_range().split())
    finally:
        pdf.close()


def test_duplex_translation_skips_tinted_chapter_page():
    pages = _make_duplex_book(n_pages=30, parity=0)
    p4 = pages[4]
    p4.page_number = 5
    p4.page_number_source = "inferred"
    p4.margin.nombre_box = None
    p4.margin.nombre_source = None
    p4.margin.bg_tint_frac = 0.90
    p4.margin.content = Box(1270, 100, 1620, 1500)
    base_crop = Box(580, -470, 2180, 1930)
    p4.margin.crop = Box(base_crop.x0, base_crop.y0, base_crop.x1, base_crop.y1)
    p4.regions = [Region(kind=RegionKind.TEXT, box=Box(1420, 100, 1530, 1400), ocr_text="第Ⅳ章")]

    pairing = determine_duplex_pairing(pages)
    assert pairing.known
    res = apply_duplex_translation_exceptions(pages, pairing)
    assert 4 not in res["partner"]
    assert 4 not in res["median"]
    assert pages[4].margin.crop == base_crop


def test_predicted_folio_protects_ink_from_margin_fill():
    import cv2
    from hokusai_press.geometry.margin import detect_predicted_folio_box, protect_edge_folio
    from hokusai_press.model import RenderSettings
    from hokusai_press.render import render_page_image

    pages = _make_duplex_book(n_pages=10, parity=0)
    # Reliable even-indexed pages have nombre_box at (100, 2800, 140, 2830) relative to (200, 300) crop
    for k in range(0, 10, 2):
        pages[k].margin.crop = Box(200, 300, 1800, 2700)
        pages[k].margin.nombre_box = Box(300, 2600, 340, 2630)
        pages[k].margin.nombre_source = "folio"

    # Target page 4: folio_not_found, content excludes folio at (305, 2605, 338, 2627)
    p4 = pages[4]
    p4.page_number = None
    p4.page_number_source = None
    p4.margin.crop = Box(200, 300, 1800, 2700)
    p4.margin.nombre_box = None
    p4.margin.nombre_source = None
    p4.margin.content = Box(400, 350, 1750, 2650)

    img = np.full((3000, 2000, 3), 255, dtype=np.uint8)
    cv2.putText(img, "13", (305, 2625), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

    ef = detect_predicted_folio_box(img, pages, 4, dpi=300.0)
    assert ef is not None
    protect_edge_folio(p4, ef)
    assert p4.margin.edge_folio_box is not None

    out, _, _, _ = render_page_image(img, p4, RenderSettings(target_dpi=300))
    # Folio pixels inside output coordinates (around x=105..140, y=2300..2330) must NOT be whitened
    patch = out[2295:2335, 100:145]
    assert patch.min() < 100


def test_predicted_folio_noop_when_no_ink():
    from hokusai_press.geometry.margin import detect_predicted_folio_box

    pages = _make_duplex_book(n_pages=10, parity=0)
    for k in range(0, 10, 2):
        pages[k].margin.crop = Box(200, 300, 1800, 2700)
        pages[k].margin.nombre_box = Box(300, 2600, 340, 2630)
        pages[k].margin.nombre_source = "folio"

    p4 = pages[4]
    p4.page_number = None
    p4.page_number_source = None
    p4.margin.nombre_box = None
    p4.margin.nombre_source = None
    p4.margin.content = Box(400, 350, 1750, 2650)

    blank_img = np.full((3000, 2000, 3), 255, dtype=np.uint8)
    ef = detect_predicted_folio_box(blank_img, pages, 4, dpi=300.0)
    assert ef is None
    assert p4.margin.edge_folio_box is None


@pytest.mark.parametrize("skipped", [False, True])
def test_supported_internal_blank_keeps_source_indices_and_restores_sheet_parity(skipped):
    pages = _make_duplex_book(n_pages=40, parity=0)
    if skipped:
        del pages[15]
        for i, page in enumerate(pages):
            page.source.page_index = i

    before = [page.margin.crop for page in pages]
    pairing = determine_duplex_pairing(pages)
    result = detect_blank_insertions(pages, pairing)
    assert result["insert_before"] == ([15] if skipped else [])
    slots = logical_page_indices(pages, result["insert_before"])
    assert [(slot + 1) % 2 for slot in slots] == [p.page_number % 2 for p in pages]
    assert [p.source.page_index for p in pages] == list(range(len(pages)))

    corrected = determine_duplex_pairing(pages, logical_indices=slots)
    assert corrected.known
    assert corrected.partner_of(13) == 12
    assert corrected.partner_of(14) == (None if skipped else 15)
    assert corrected.partner_of(15) == (16 if skipped else 14)

    if not skipped:
        assert align_after_blank_insertions(pages, result) == []
        assert [page.margin.crop for page in pages] == before


def test_recurring_divider_locates_gap_before_unnumbered_title_page():
    pages = _make_duplex_book(n_pages=46, parity=0)
    for p in pages:
        if p.page_number % 2:
            p.margin.content = Box(280, 390, 1810, 2700)
            p.margin.nombre_box = Box(320, 2660, 360, 2690)
        else:
            p.margin.content = Box(180, 390, 1710, 2700)
            p.margin.nombre_box = Box(1640, 2660, 1680, 2690)
    for i in (8, 18):
        p = pages[i]
        p.page_number = None
        p.page_number_source = None
        p.margin.nombre_box = None
        p.margin.nombre_source = None
        p.margin.content = Box(440, 480, 485, 1300)
        p.regions = [Region(RegionKind.TEXT, p.margin.content, ocr_text="章扉")]
    del pages[7]
    for i, p in enumerate(pages):
        p.source.page_index = i

    pairing = determine_duplex_pairing(pages)
    assert pairing.known
    result = detect_blank_insertions(pages, pairing)
    assert result["gaps"] == {7: "chapter_divider"}
    assert result["insert_before"] == [7]
    slots = logical_page_indices(pages, result["insert_before"])
    assert slots[7] == 8
    assert slots[17] == 18
