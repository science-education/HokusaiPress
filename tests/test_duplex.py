"""Unit tests for duplex sheet-pair alignment, skew, and blank-page insertion."""

from __future__ import annotations

import numpy as np
import pypdfium2 as pdfium
import pytest

from hokusai_press.duplex import (
    apply_duplex_skew_exceptions,
    apply_duplex_translation_exceptions,
    detect_blank_insertions,
    determine_duplex_pairing,
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

    # Create synthetic front (tilted +5 deg, deskewed at -5 -> text horizontal when rotated -5)
    # and back (tilted -5 deg, deskewed at +5, PLUS a strong vertical scanner streak at 0 deg!)
    h, w = 1000, 700
    base = np.full((h, w, 3), 255, dtype=np.uint8)
    for y in range(150, 850, 40):
        cv2.line(base, (100, y), (600, y), (0, 0, 0), 3)

    M_front = cv2.getRotationMatrix2D((w / 2, h / 2), -5.0, 1.0)
    img_front = cv2.warpAffine(base, M_front, (w, h), borderValue=(255, 255, 255))

    M_back = cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 1.0)
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
