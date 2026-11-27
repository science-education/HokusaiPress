"""Unit tests for full-sheet tinted page detection, placement, and rendering."""

import numpy as np
import pikepdf

from hokusai_press.duplex import apply_duplex_translation_exceptions, determine_duplex_pairing
from hokusai_press.geometry.margin import (
    compute_page_tint_stats,
    detect_tinted_pages,
    normalize_margins,
    normalize_margins_pass1,
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
    RenderSettings,
    SourceRef,
)
from hokusai_press.render import build_pdf, render_output_preview, render_page_image


def _make_body_img(h: int = 800, w: int = 600, bg: int = 255) -> np.ndarray:
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    # Draw sparse thin text-like strokes in the interior (realistic ink density)
    for y in range(100, 700, 40):
        img[y:y + 1, 80:520] = 20
    return img


def _make_halftone_img(h: int = 800, w: int = 600, dot_val: int = 190) -> np.ndarray:
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Checkerboard halftone across the entire sheet
    yy, xx = np.ogrid[:h, :w]
    mask = ((yy // 2 + xx // 2) % 2) == 0
    img[mask] = dot_val
    # Title text in a small box on the right
    img[200:500, 420:460] = 20
    return img


def _make_partial_tint_img(h: int = 800, w: int = 600, tint_val: int = 200) -> np.ndarray:
    img = _make_body_img(h, w, bg=255)
    # Shaded box covering ~50% of the sheet
    img[150:600, 80:500] = tint_val
    return img


def _page(idx: int, content: Box, page_w: int = 600, page_h: int = 800,
          blank: bool = False, tinted: bool = False,
          page_number: int | None = None, nombre_box: Box | None = None) -> PageParams:
    return PageParams(
        source=SourceRef(path="dummy.pdf", page_index=idx),
        dpi=300.0,
        deskew=Deskew(angle_deg=0.0, confidence=1.0),
        margin=Margin(
            content=content,
            confidence=1.0,
            page_w=float(page_w),
            page_h=float(page_h),
            tinted=tinted,
            nombre_box=nombre_box,
            nombre_source="folio" if nombre_box is not None else None,
        ),
        page_number=page_number,
        page_number_source="folio" if page_number is not None else None,
        blank=blank,
    )


def test_detect_tinted_pages_halftone_blank_body_partial():
    body_imgs = [_make_body_img() for _ in range(5)]
    halftone_img = _make_halftone_img(dot_val=190)  # mean ~222.5
    blank_img = np.full((800, 600, 3), 180, dtype=np.uint8)  # dark scan marked blank
    partial_img = _make_partial_tint_img()

    imgs = body_imgs + [halftone_img, blank_img, partial_img]
    pages = [
        _page(i, Box(80, 100, 520, 700), blank=(i == 6))
        for i in range(len(imgs))
    ]
    stats = [compute_page_tint_stats(im) for im in imgs]
    flags = detect_tinted_pages(pages, stats)

    assert flags[:5] == [False] * 5   # body pages: not tinted
    assert flags[5] is True            # full-sheet halftone: tinted
    assert flags[6] is False           # blank page: excluded
    assert flags[7] is False           # partial tint box: not tinted


def test_tinted_page_placement_fill_and_no_whiteout():
    # Body pages define target_w/h around 500x700; tinted sheet is 600x800
    # -> within FULL_BLEED_MAX_CUT? Wait: 600/500 = 1.2 > 1.1, so make body 580x780
    pages = [
        _page(i, Box(30, 30, 550, 750), page_w=600, page_h=800,
              page_number=i + 1, nombre_box=Box(280, 760, 320, 775))
        for i in range(6)
    ]
    # Tinted page with small title content on the right
    tinted_p = _page(6, Box(420, 200, 460, 500), page_w=600, page_h=800, tinted=True)
    pages.append(tinted_p)

    normalize_margins_pass1(pages, output_margin_mm=5.0)
    assert tinted_p.margin.fill is True
    assert Flag.CONTENT_SCALED_DOWN not in tinted_p.flags
    assert tinted_p.margin.crop is not None
    # Crop should be centered sheet trim window, not tight around title
    assert tinted_p.margin.crop.width > 500
    assert tinted_p.margin.crop.height > 700

    # Render must NOT whiteout outside content box and must emit full-page photo layer
    halftone = _make_halftone_img(800, 600, dot_val=190)
    rs = RenderSettings(target_dpi=300)
    out_bgr, _, mode, photo_boxes = render_page_image(halftone, tinted_p, rs)
    assert mode == "bw"
    assert len(photo_boxes) == 1
    assert photo_boxes[0][:4] == (0.0, 0.0, float(out_bgr.shape[1]), float(out_bgr.shape[0]))
    # Top-left corner (outside title content Box(420,200,460,500)) must retain halftone tint, not 255 white
    assert float(out_bgr[:40, :40].mean()) < 245.0

    # Preview must also retain halftone tint (not binarized or whitened)
    prev = render_output_preview(halftone, tinted_p, rs)
    assert float(prev[:40, :40].mean()) < 245.0


def test_tinted_page_oversize_fallback_keeps_whole_sheet():
    # Body pages have small target size (300x400), tinted sheet is 600x800 (ratio 2.0 > FULL_BLEED_MAX_CUT)
    pages = [
        _page(i, Box(50, 50, 250, 350), page_w=300, page_h=400)
        for i in range(6)
    ]
    tinted_p = _page(6, Box(420, 200, 460, 500), page_w=600, page_h=800, tinted=True)
    pages.append(tinted_p)

    normalize_margins_pass1(pages, output_margin_mm=5.0)
    assert tinted_p.margin.fill is False
    assert Flag.CONTENT_SCALED_DOWN in tinted_p.flags
    assert tinted_p.margin.crop.width >= 600.0
    assert tinted_p.margin.crop.height >= 800.0


def test_tinted_page_excluded_from_pass2_and_part_b():
    pages = []
    for i in range(10):
        # Alternating parity folio offsets
        x0 = 60.0 if i % 2 == 0 else 500.0
        pages.append(
            _page(i, Box(80, 100, 520, 700), page_w=600, page_h=800,
                  page_number=i + 1, nombre_box=Box(x0, 750, x0 + 30, 770))
        )
    # Page 4 is tinted (even though it has no trusted folio)
    pages[4].page_number = None
    pages[4].page_number_source = None
    pages[4].margin.nombre_box = None
    pages[4].margin.nombre_source = None
    pages[4].margin.tinted = True
    pages[4].margin.content = Box(420, 200, 460, 500)

    normalize_margins(pages, output_margin_mm=5.0)
    crop_before_b = Box(
        pages[4].margin.crop.x0, pages[4].margin.crop.y0,
        pages[4].margin.crop.x1, pages[4].margin.crop.y1,
    )
    pairing = determine_duplex_pairing(pages)
    res = apply_duplex_translation_exceptions(pages, pairing)
    assert 4 not in res["partner"]
    assert 4 not in res["median"]
    assert 4 not in res["uncorrected"]
    assert pages[4].margin.crop == crop_before_b


def test_tinted_page_pdf_keeps_text_layer_and_gray_image(tmp_path):
    halftone = _make_halftone_img(400, 300, dot_val=190)
    p = _page(0, Box(200, 100, 240, 250), page_w=300, page_h=400, tinted=True)
    p.regions = [Region(kind=RegionKind.TEXT, box=Box(200, 100, 240, 250), ocr_text="第1章")]
    normalize_margins_pass1([p], output_margin_mm=5.0)
    doc = Document(source_path="dummy.pdf", pages=[p], render=RenderSettings(target_dpi=300))
    out_pdf = str(tmp_path / "tinted.pdf")
    build_pdf(doc, [halftone], out_pdf)

    with pikepdf.open(out_pdf) as pdf:
        assert len(pdf.pages) == 1
        # Must contain both the bilevel base and the photo/gray overlay XObject
        xobjs = pdf.pages[0].Resources.get("/XObject", {})
        assert len(xobjs) >= 2
