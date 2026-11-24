"""Tests for 3-pass folio-aligned margin normalization (Pass 2a/2b/3) and halftone handling."""

from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest

from hokusai_press.geometry.margin import normalize_margins, normalize_margins_pass1
from hokusai_press.model import Box, Deskew, Flag, Margin, PageParams, Region, RegionKind, SourceRef
from hokusai_press.nombre_reader import _detect_band_tokens, localize_folios_guided


DPI = 300.0
PX_PER_MM = DPI / 25.4


def _mm(v: float) -> float:
    return v * PX_PER_MM


def _to_mm(v: float) -> float:
    return v / PX_PER_MM


def _make_yokogaki_book_page(
    idx: int,
    *,
    body_dx_mm: float = 0.0,
    body_dy_mm: float = 0.0,
    nb_outer_x_mm: float | None = None,
    nb_y_mm: float = 192.0,
    nb_w_mm: float = 4.0,
    nb_h_mm: float = 3.0,
    page_number: int | None = None,
    nombre_source: str | None = "folio",
    centered: bool = False,
    paper_shift_x_mm: float = 0.0,
    paper_shift_y_mm: float = 0.0,
    extra_content_x1_mm: float | None = None,
    blank: bool = False,
) -> PageParams:
    """Create a realistic 140x210mm yokogaki page at 300 dpi.

    Body lines sit inside the sheet with optional scan shift (`paper_shift_*`)
    and optional typography start-corner shift (`body_dx_mm`, `body_dy_mm`, e.g.
    1-line chapter drop or indent while the printed folio stays fixed on paper).
    """
    pw, ph = _mm(140.0), _mm(210.0)
    # Body lines: 18 horizontal lines, 100mm wide
    bx0 = _mm(20.0 + paper_shift_x_mm + body_dx_mm)
    by0 = _mm(22.0 + paper_shift_y_mm + body_dy_mm)
    bx1 = _mm(120.0 + paper_shift_x_mm)
    lines = [
        Box(bx0, by0 + _mm(k * 8.0), bx1, by0 + _mm(k * 8.0 + 3.5))
        for k in range(18)
    ]
    regions = [Region(kind=RegionKind.TEXT, box=b, ocr_text="本文行") for b in lines]

    nb_box = None
    if nb_outer_x_mm is not None:
        ny0 = _mm(nb_y_mm + paper_shift_y_mm - nb_h_mm / 2.0)
        ny1 = _mm(nb_y_mm + paper_shift_y_mm + nb_h_mm / 2.0)
        if centered:
            ncx = _mm(nb_outer_x_mm + paper_shift_x_mm)
            nb_box = Box(ncx - _mm(nb_w_mm / 2.0), ny0, ncx + _mm(nb_w_mm / 2.0), ny1)
        elif idx % 2 == 0:
            # Recto (even 0-based index): outer edge = right (x1)
            nx1 = _mm(nb_outer_x_mm + paper_shift_x_mm)
            nb_box = Box(nx1 - _mm(nb_w_mm), ny0, nx1, ny1)
        else:
            # Verso (odd 0-based index): outer edge = left (x0)
            nx0 = _mm(nb_outer_x_mm + paper_shift_x_mm)
            nb_box = Box(nx0, ny0, nx0 + _mm(nb_w_mm), ny1)

    xs0 = [b.x0 for b in lines] + ([nb_box.x0] if nb_box else [])
    ys0 = [b.y0 for b in lines] + ([nb_box.y0] if nb_box else [])
    xs1 = [b.x1 for b in lines] + ([nb_box.x1] if nb_box else [])
    ys1 = [b.y1 for b in lines] + ([nb_box.y1] if nb_box else [])
    if extra_content_x1_mm is not None:
        xs1.append(_mm(extra_content_x1_mm + paper_shift_x_mm))
    content = Box(min(xs0), min(ys0), max(xs1), max(ys1))

    return PageParams(
        source=SourceRef(path="synthetic.pdf", page_index=idx),
        dpi=DPI,
        deskew=Deskew(0.0, 1.0),
        margin=Margin(
            content=content,
            confidence=0.95,
            nombre_box=nb_box,
            nombre_source=nombre_source if nb_box is not None else None,
            page_w=pw,
            page_h=ph,
        ),
        regions=regions,
        page_number=page_number,
        nombre_text=str(page_number) if page_number is not None else None,
        blank=blank,
    )


def test_outer_edge_alignment_across_digit_widths_and_shared_y():
    """1-digit (8), 2-digit (48), 3-digit (128) folios with different widths
    must align on the fore-edge outer boundary (right x1 on recto, left x0 on
    verso) and share one common vertical height across odd and even parities."""
    pages: list[PageParams] = []
    # 20 pages with body start-corner jitter + parity height difference in body
    widths_mm = {0: 2.0, 1: 2.0, 4: 4.0, 5: 4.0, 14: 6.5, 15: 6.5}  # 1/2/3-digit widths
    for i in range(20):
        # Introduce body start-corner variation (e.g. chapter drop / indent / ADF shift)
        # plus systematic 4mm Y difference in body start between odd and even pages!
        body_dx = (2.5 if i % 4 == 0 else (-2.0 if i % 5 == 0 else 0.0))
        body_dy = (4.0 if i % 2 == 1 else 0.0) + (3.0 if i in (4, 15) else 0.0)
        # Also include an unanchored page (exceeds ANCHOR_TOL_IN = 5.08mm)
        paper_dx = 7.0 if i == 10 else ((i % 3) - 1) * 1.5
        nb_outer = 125.0 if i % 2 == 0 else 15.0
        nb_w = widths_mm.get(i, 4.0)
        pages.append(
            _make_yokogaki_book_page(
                i,
                body_dx_mm=body_dx,
                body_dy_mm=body_dy,
                nb_outer_x_mm=nb_outer,
                nb_y_mm=192.0,
                nb_w_mm=nb_w,
                page_number=8 + i,
                paper_shift_x_mm=paper_dx,
            )
        )

    normalize_margins(pages, output_margin_mm=5.0)

    even_x1 = [_to_mm(p.margin.nombre_box.x1 - p.margin.crop.x0) for p in pages if p.source.page_index % 2 == 0]
    odd_x0 = [_to_mm(p.margin.nombre_box.x0 - p.margin.crop.x0) for p in pages if p.source.page_index % 2 == 1]
    all_cy = [
        _to_mm((p.margin.nombre_box.y0 + p.margin.nombre_box.y1) / 2.0 - p.margin.crop.y0)
        for p in pages
    ]

    # Outer edge matches across 1-, 2-, and 3-digit pages (< 0.1 mm spread)
    assert max(even_x1) - min(even_x1) < 0.1
    assert max(odd_x0) - min(odd_x0) < 0.1
    # Vertical height is shared across odd and even parities (< 0.1 mm spread)
    assert max(all_cy) - min(all_cy) < 0.1


def test_centered_nombre_aligns_on_center():
    pages = [
        _make_yokogaki_book_page(
            i,
            body_dx_mm=(3.0 if i % 3 == 0 else -1.5),
            body_dy_mm=(2.5 if i % 2 == 1 else 0.0),
            nb_outer_x_mm=70.0,
            nb_y_mm=192.0,
            nb_w_mm=(2.2 if i < 6 else 5.8),
            page_number=1 + i,
            centered=True,
        )
        for i in range(18)
    ]
    normalize_margins(pages, output_margin_mm=5.0)

    cxs = [
        _to_mm((p.margin.nombre_box.x0 + p.margin.nombre_box.x1) / 2.0 - p.margin.crop.x0)
        for p in pages
    ]
    cys = [
        _to_mm((p.margin.nombre_box.y0 + p.margin.nombre_box.y1) / 2.0 - p.margin.crop.y0)
        for p in pages
    ]
    assert max(cxs) - min(cxs) < 0.1
    assert max(cys) - min(cys) < 0.1


def test_bad_nombre_poison_pages_do_not_move_page():
    """Misidentified folios (stray dirt, forged sequential running-head kanji,
    fused wide running-head box, and 0010 p128 / 0012 p7 style large offsets)
    must be rejected with 0.00 mm movement from Pass 1 and 0 clips."""
    base_pages = [
        _make_yokogaki_book_page(
            i,
            nb_outer_x_mm=(125.0 if i % 2 == 0 else 15.0),
            nb_y_mm=192.0,
            page_number=10 + i,
        )
        for i in range(20)
    ]

    # Poison 1 (page 4): stray mark / dirt near top-left, no sequence match
    base_pages[4].margin.nombre_box = Box(_mm(35.0), _mm(12.0), _mm(38.0), _mm(15.0))
    base_pages[4].page_number = None
    # Poison 2 (page 6): forged sequential running-head kanji at top band!
    base_pages[6].margin.nombre_box = Box(_mm(120.0), _mm(10.0), _mm(126.0), _mm(14.0))
    base_pages[6].page_number = 16  # matches dominant offset +10!
    # Poison 3 (page 8): running-head + folio fused into a wide box (25mm wide)
    base_pages[8].margin.nombre_box = Box(_mm(100.0), _mm(190.5), _mm(125.0), _mm(193.5))
    base_pages[8].page_number = 18
    # Poison 4 (page 10): forged sequential box 15mm off on an anchored page (0010 p128 / 0012 p7 type)
    base_pages[10].margin.nombre_box = Box(_mm(106.0), _mm(190.5), _mm(110.0), _mm(193.5))
    base_pages[10].page_number = 20

    pass1_pages = copy.deepcopy(base_pages)
    normalize_margins_pass1(pass1_pages, output_margin_mm=5.0)

    normalize_margins(base_pages, output_margin_mm=5.0)

    for pidx in (4, 6, 8, 10):
        c1 = pass1_pages[pidx].margin.crop
        c2 = base_pages[pidx].margin.crop
        assert abs(c2.x0 - c1.x0) < 1e-6
        assert abs(c2.y0 - c1.y0) < 1e-6
        assert abs(c2.x1 - c1.x1) < 1e-6
        assert abs(c2.y1 - c1.y1) < 1e-6

    # No page clips its content
    for p in base_pages:
        crop, c = p.margin.crop, p.margin.content
        assert crop.x0 <= c.x0 + 1e-6 and crop.x1 >= c.x1 - 1e-6
        assert crop.y0 <= c.y0 + 1e-6 and crop.y1 >= c.y1 - 1e-6


def test_no_trusted_nombre_or_geom_only_identical_to_pass1():
    """Books with only shape-only ('geom') folios, scattered legacy garbage,
    or too few trusted folios must produce output identical to Pass 1."""
    # Case 1: All pages have neatly aligned _detect_nombre ("geom") boxes -> MUST NOT activate
    geom_pages = [
        _make_yokogaki_book_page(
            i,
            body_dx_mm=(3.0 if i % 2 == 0 else -3.0),
            nb_outer_x_mm=(125.0 if i % 2 == 0 else 15.0),
            nb_y_mm=192.0,
            page_number=10 + i,
            nombre_source="geom",
        )
        for i in range(20)
    ]
    ref_geom = copy.deepcopy(geom_pages)
    normalize_margins_pass1(ref_geom, output_margin_mm=5.0)
    normalize_margins(geom_pages, output_margin_mm=5.0)
    for p, r in zip(geom_pages, ref_geom):
        assert p.margin.crop == r.margin.crop

    # Case 2: Corrupted legacy DB (nombre_source=None, page_number set, boxes scattered in running head)
    corrupt_pages = [
        _make_yokogaki_book_page(
            i,
            nb_outer_x_mm=float(30 + (i * 23) % 80),
            nb_y_mm=(10.0 if i % 3 == 0 else 192.0),
            page_number=10 + i,
            nombre_source=None,
            centered=True,
        )
        for i in range(20)
    ]
    ref_corrupt = copy.deepcopy(corrupt_pages)
    normalize_margins_pass1(ref_corrupt, output_margin_mm=5.0)
    normalize_margins(corrupt_pages, output_margin_mm=5.0)
    for p, r in zip(corrupt_pages, ref_corrupt):
        assert p.margin.crop == r.margin.crop

    # Case 3: Too few trusted folios (only 4 pages out of 20)
    sparse_pages = [
        _make_yokogaki_book_page(
            i,
            body_dx_mm=(2.5 if i % 2 == 0 else -2.5),
            nb_outer_x_mm=(125.0 if i % 2 == 0 else 15.0) if i < 4 else None,
            nb_y_mm=192.0,
            page_number=(10 + i) if i < 4 else None,
        )
        for i in range(20)
    ]
    ref_sparse = copy.deepcopy(sparse_pages)
    normalize_margins_pass1(ref_sparse, output_margin_mm=5.0)
    normalize_margins(sparse_pages, output_margin_mm=5.0)
    for p, r in zip(sparse_pages, ref_sparse):
        assert p.margin.crop == r.margin.crop


def test_no_clip_no_scale_fill_blank_and_idempotent():
    """Verify no-clip clamp, zero scaling (crop_w/h unchanged), fill/blank/
    unnumbered pages remain at Pass 1, and repeated calls are bit-for-bit idempotent."""
    pages = [
        _make_yokogaki_book_page(
            i,
            body_dx_mm=(3.5 if i % 3 == 0 else -2.0),
            body_dy_mm=(2.0 if i % 2 == 1 else 0.0),
            nb_outer_x_mm=(125.0 if i % 2 == 0 else 15.0),
            nb_y_mm=192.0,
            page_number=10 + i,
            extra_content_x1_mm=(136.0 if i == 6 else None),  # thumb index near edge
        )
        for i in range(20)
    ]
    # Add a blank page, an unnumbered divider, and a full-bleed black sheet
    pages[18].blank = True
    pages[19].margin.nombre_box = None
    pages[19].margin.nombre_source = None
    pages[19].page_number = None
    # Full-bleed page larger than book target by 4%
    fb = _make_yokogaki_book_page(20, nb_outer_x_mm=None)
    fb.margin.content = Box(0.0, 0.0, _mm(145.0), _mm(216.0))
    fb.margin.page_w = _mm(145.0)
    fb.margin.page_h = _mm(216.0)
    pages.append(fb)

    pass1 = copy.deepcopy(pages)
    normalize_margins_pass1(pass1, output_margin_mm=5.0)

    normalize_margins(pages, output_margin_mm=5.0)
    crops_first = [copy.deepcopy(p.margin.crop) for p in pages]

    # Idempotency: second call must produce bit-for-bit identical crops
    normalize_margins(pages, output_margin_mm=5.0)
    for p, c_first in zip(pages, crops_first):
        assert p.margin.crop == c_first

    # Blank, unnumbered, and fill pages stay at exact Pass-1 crop
    assert pages[20].margin.fill is True
    for idx in (18, 19, 20):
        assert pages[idx].margin.crop == pass1[idx].margin.crop

    # Zero scaling: crop width & height identical to Pass 1, no new CONTENT_SCALED_DOWN
    for p, p1 in zip(pages, pass1):
        assert abs(p.margin.crop.width - p1.margin.crop.width) < 1e-6
        assert abs(p.margin.crop.height - p1.margin.crop.height) < 1e-6
        assert (Flag.CONTENT_SCALED_DOWN in p.flags) == (Flag.CONTENT_SCALED_DOWN in p1.flags)
        # Strict no-clip on every non-fill page
        if not p.margin.fill:
            crop, c = p.margin.crop, p.margin.content
            assert crop.x0 <= c.x0 + 1e-6 and crop.x1 >= c.x1 - 1e-6
            assert crop.y0 <= c.y0 + 1e-6 and crop.y1 >= c.y1 - 1e-6


def test_detect_band_tokens_extracts_digits_on_halftone_screen():
    """Direct unit test for halftone/screened (網掛け) ROI re-binarization:
    a light-gray screened tint panel (~215-225) behind dark digits (~50) merges
    into one oversized component under 'bg - 38', and must split cleanly."""
    px_per_mm = 150.0 / 25.4
    bh, bw = int(round(30 * px_per_mm)), int(round(140 * px_per_mm))
    band = np.full((bh, bw), 255, dtype=np.uint8)

    # Draw a 55mm x 28mm light-gray halftone panel on the right half (gray ~212..224)
    rng = np.random.default_rng(42)
    sx0 = int(round(80 * px_per_mm))
    sx1 = int(round(135 * px_per_mm))
    panel = rng.integers(205, 226, size=(bh - 2, sx1 - sx0), dtype=np.uint8)
    band[1:bh - 1, sx0:sx1] = panel

    # Draw crisp dark folio digits "152" on top of the halftone panel
    fx = int(round(120 * px_per_mm))
    fy = int(round(20 * px_per_mm))
    cv2.putText(band, "152", (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, 45, 1, cv2.LINE_AA)

    tokens = _detect_band_tokens(band, y_offset=1000, px_per_mm=px_per_mm)
    # Must find a digit-sized token right where "152" was drawn
    hits = [
        t for t in tokens
        if abs(((t[0] + t[2]) / 2.0) - (fx + 14)) < 25 and abs(((t[1] + t[3]) / 2.0) - (1000 + fy - 6)) < 15
    ]
    assert len(hits) >= 1


def test_guided_folio_localization_with_halftone_chapter_pages():
    """End-to-end synthetic Pass 1 -> Pass 2a (real folio_digit.onnx) -> Pass 2b/3
    test including halftone/screened chapter-opener pages (0006 scenario)."""
    dpi = 150.0
    pm = dpi / 25.4
    pw, ph = int(round(140 * pm)), int(round(210 * pm))
    rng = np.random.default_rng(7)

    pages: list[PageParams] = []
    originals: list[np.ndarray] = []
    screened_indices = {3, 9, 13}  # verso (odd index) chapter openers with halftone panel

    for idx in range(16):
        img = np.full((ph, pw, 3), 255, dtype=np.uint8)
        # Body start-corner jitter so Pass 1 leaves folio misaligned by ~3mm
        body_dx = 3.0 if idx % 3 == 0 else (-2.5 if idx % 4 == 0 else 0.0)
        body_dy = 2.5 if idx % 2 == 1 else 0.0
        bx0 = int(round((20.0 + body_dx) * pm))
        by0 = int(round((22.0 + body_dy) * pm))
        bx1 = int(round(120.0 * pm))
        lines = []
        for k in range(18):
            ly0 = by0 + int(round(k * 8.0 * pm))
            ly1 = ly0 + int(round(3.0 * pm))
            cv2.rectangle(img, (bx0, ly0), (bx1, ly1), (30, 30, 30), -1)
            lines.append(Box(float(bx0), float(ly0), float(bx1), float(ly1)))

        # Halftone screen covering right half of bottom margin on chapter openers
        if idx in screened_indices:
            sx0, sx1 = int(round(78 * pm)), int(round(136 * pm))
            sy0, sy1 = int(round(170 * pm)), int(round(206 * pm))
            panel = rng.integers(208, 226, size=(sy1 - sy0, sx1 - sx0), dtype=np.uint8)
            img[sy0:sy1, sx0:sx1] = panel[:, :, None]

        # Draw printed sequential folio number (1..16) at bottom outer corner
        num_str = str(idx + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.55, 1
        (tw, th), _ = cv2.getTextSize(num_str, font, scale, thick)
        fy_base = int(round(196.0 * pm))
        if idx % 2 == 0:
            # Left outer corner (x0 aligned)
            fx = int(round(14.0 * pm))
        else:
            # Right outer corner (x1 aligned)
            fx = int(round(126.0 * pm)) - tw
        cv2.putText(img, num_str, (fx, fy_base), font, scale, (35, 35, 35), thick, cv2.LINE_AA)

        # Content box includes body + folio
        content = Box(
            float(min(bx0, fx)),
            float(by0),
            float(max(bx1, fx + tw)),
            float(fy_base + 4),
        )
        p = PageParams(
            source=SourceRef(path="synth_screen.pdf", page_index=idx),
            dpi=dpi,
            deskew=Deskew(0.0, 1.0),
            margin=Margin(
                content=content,
                confidence=0.95,
                page_w=float(pw),
                page_h=float(ph),
            ),
            regions=[Region(kind=RegionKind.TEXT, box=b, ocr_text="行") for b in lines],
        )
        pages.append(p)
        originals.append(img)

    # Pass 1 -> Pass 2a -> Pass 2b/3
    normalize_margins_pass1(pages, output_margin_mm=5.0)
    res = localize_folios_guided(pages, originals=originals, max_workers=2)
    assert res["applied"] is True
    assert res["dom_off"] == 1
    assert res["seq_verified"] == 16

    # Every screened halftone chapter opener must be found with exact page_number
    for idx in screened_indices:
        p = pages[idx]
        assert p.margin.nombre_source == "folio"
        assert p.margin.nombre_box is not None
        assert p.page_number == idx + 1

    # Pass 2b/3 aligns all folios tightly
    normalize_margins(pages, output_margin_mm=5.0)
    even_x0 = [(p.margin.nombre_box.x0 - p.margin.crop.x0) / pm for p in pages if p.source.page_index % 2 == 0]
    odd_x1 = [(p.margin.nombre_box.x1 - p.margin.crop.x0) / pm for p in pages if p.source.page_index % 2 == 1]
    all_cy = [((p.margin.nombre_box.y0 + p.margin.nombre_box.y1) / 2.0 - p.margin.crop.y0) / pm for p in pages]
    assert max(even_x0) - min(even_x0) < 0.1
    assert max(odd_x1) - min(odd_x1) < 0.1
    assert max(all_cy) - min(all_cy) < 0.1


def test_cli_no_nombre_align_option(tmp_path, monkeypatch):
    from PIL import Image
    from hokusai_press import cli as cli_mod
    from hokusai_press import pipeline as pipeline_mod

    src = str(tmp_path / "sample.pdf")
    out = str(tmp_path / "out.pdf")
    db = str(tmp_path / "test.db")
    Image.fromarray(np.full((200, 150, 3), 255, np.uint8)).save(src, resolution=100)

    seen_run = []
    seen_remargin = []

    def fake_run(*args, **kwargs):
        seen_run.append(kwargs.get("align_nombre"))
        return {"doc_id": "sample.pdf", "pages": 1, "needs_review": [], "out_pdf": out, "warnings": [], "profile": {}}

    def fake_remargin(*args, **kwargs):
        seen_remargin.append(kwargs.get("align_nombre"))
        return 1

    def fake_rebuild(*args, **kwargs):
        return {"doc_id": "sample.pdf", "pages": 1, "out_pdf": out}

    monkeypatch.setattr(pipeline_mod, "run", fake_run)
    monkeypatch.setattr(pipeline_mod, "recompute_shadows_and_margins", fake_remargin)
    monkeypatch.setattr(pipeline_mod, "rebuild", fake_rebuild)

    assert cli_mod.main(["run", src, "--out", out, "--db", db, "--no-ocr", "--no-nombre-align"]) == 0
    assert cli_mod.main(["run", src, "--out", out, "--db", db, "--no-ocr"]) == 0
    assert seen_run == [False, True]

    assert cli_mod.main(["remargin", src, "--out", out, "--db", db, "--no-nombre-align"]) == 0
    assert cli_mod.main(["remargin", src, "--out", out, "--db", db]) == 0
    assert seen_remargin == [False, True]


def test_clear_unlocated_nombre_drops_stale_arabic_keeps_roman():
    from hokusai_press.nombre_reader import _clear_unlocated_nombre

    def mk(num, txt):
        p = PageParams(source=SourceRef(path="x.pdf", page_index=0))
        p.margin = Margin(content=Box(0, 0, 10, 10), nombre_box=Box(1, 1, 2, 2))
        p.page_number, p.nombre_text = num, txt
        return p

    stale = mk(3, "3")
    _clear_unlocated_nombre(stale)
    assert stale.page_number is None and stale.nombre_text is None
    assert stale.margin.nombre_box is None and stale.margin.nombre_source is None

    roman = mk(4, "iv")
    _clear_unlocated_nombre(roman)
    assert roman.page_number == 4 and roman.nombre_text == "iv"


def test_protected_box_ignores_bleed_regions_and_plain_ink():
    from hokusai_press.geometry.margin import _protected_box

    dpi = 254.0  # 10 px/mm
    p = PageParams(source=SourceRef(path="x.pdf", page_index=0))
    p.margin = Margin(content=Box(0, 0, 1000, 1500), page_w=1000, page_h=1500,
                      nombre_box=Box(100, 1400, 140, 1430))
    p.regions = [
        Region(box=Box(200, 200, 800, 1300), kind=RegionKind.TEXT),
        Region(box=Box(5, 50, 300, 80), kind=RegionKind.TEXT),  # bleed tab bar
    ]
    pb = _protected_box(p, dpi)
    assert (pb.x0, pb.y0, pb.x1, pb.y1) == (100, 200, 800, 1430)

    p.regions = []
    p.margin.nombre_box = None
    assert _protected_box(p, dpi) == p.margin.content
