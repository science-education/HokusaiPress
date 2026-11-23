import gc
import json
from dataclasses import asdict

import cv2
import numpy as np

from hokusai_press import pipeline as pipeline_mod
from hokusai_press.geometry import margin as margin_mod
from hokusai_press.model import (
    Box,
    Deskew,
    Flag,
    Margin,
    PageParams,
    Region,
    RegionKind,
    ReviewStatus,
    SourceRef,
    _enc,
)
from hokusai_press.pipeline import recompute_margins, recompute_shadows_and_margins
from hokusai_press.store import Store


def _make_page(
    path: str,
    idx: int,
    content: Box,
    *,
    flags: list[Flag] | None = None,
    review_status: ReviewStatus = ReviewStatus.AUTO,
    confidence: float = 0.9,
    page_w: float = 1000.0,
    page_h: float = 1400.0,
) -> PageParams:
    return PageParams(
        source=SourceRef(path=path, page_index=idx),
        dpi=600.0,
        deskew=Deskew(0.0, 1.0),
        margin=Margin(
            content=content,
            confidence=confidence,
            page_w=page_w,
            page_h=page_h,
        ),
        regions=[
            Region(kind=RegionKind.TEXT, box=content, ocr_text="body", ocr_conf=0.9)
        ],
        flags=list(flags or []),
        review_status=review_status,
    )


def test_recompute_margins_syncs_review_status_and_preserves_human_decisions(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = Store(db_path)
    doc_id = "book.pdf"
    try:
        # Bulk normal pages + outlier/resolved/human-decided pages so the
        # outlier exceeds both the type-area median and P97.5.
        pages = [
            _make_page(doc_id, i, Box(100, 100, 500, 900), review_status=ReviewStatus.AUTO)
            for i in range(42)
        ]
        # 42: Outlier starts AUTO, will get CONTENT_SCALED_DOWN -> NEEDS_REVIEW
        pages.append(_make_page(doc_id, 42, Box(100, 100, 1150, 1550), review_status=ReviewStatus.AUTO, page_w=2000.0, page_h=2400.0))
        # 43: Resolved: had CONTENT_SCALED_DOWN + NEEDS_REVIEW, now normal -> AUTO
        pages.append(_make_page(
            doc_id, 43, Box(100, 100, 500, 900),
            flags=[Flag.CONTENT_SCALED_DOWN],
            review_status=ReviewStatus.NEEDS_REVIEW,
        ))
        # 44: Resolved margin flag, but SCANNER_STREAK remains -> stays NEEDS_REVIEW
        pages.append(_make_page(
            doc_id, 44, Box(100, 100, 500, 900),
            flags=[Flag.CONTENT_SCALED_DOWN, Flag.SCANNER_STREAK],
            review_status=ReviewStatus.NEEDS_REVIEW,
        ))
        # 45: Human-decided APPROVED with outlier -> stays APPROVED
        pages.append(_make_page(doc_id, 45, Box(100, 100, 1150, 1550), review_status=ReviewStatus.APPROVED, page_w=2000.0, page_h=2400.0))
        # 46: Human-decided CORRECTED with no flags -> stays CORRECTED
        pages.append(_make_page(doc_id, 46, Box(100, 100, 500, 900), review_status=ReviewStatus.CORRECTED))

        for p in pages:
            store.upsert_page(doc_id, p.source.page_index, p)

        n = recompute_margins(store, doc_id)
        assert n == len(pages)

        by_idx = {r.page_index: r for r in store.list_pages(doc_id)}
        assert Flag.CONTENT_SCALED_DOWN in by_idx[42].params.flags
        assert by_idx[42].review_status == "needs_review"

        assert Flag.CONTENT_SCALED_DOWN not in by_idx[43].params.flags
        assert by_idx[43].review_status == "auto"

        assert by_idx[44].params.flags == [Flag.SCANNER_STREAK]
        assert by_idx[44].review_status == "needs_review"

        assert Flag.CONTENT_SCALED_DOWN in by_idx[45].params.flags
        assert by_idx[45].review_status == "approved"

        assert by_idx[46].params.flags == []
        assert by_idx[46].review_status == "corrected"
    finally:
        store.close()


def _write_synthetic_pdf(pdf_path: str, page_boxes: list[tuple[int, int, int, int]]) -> None:
    from PIL import Image

    imgs = []
    for x0, y0, x1, y1 in page_boxes:
        img = np.full((700, 500, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
        imgs.append(Image.fromarray(img))
    imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:], resolution=100)


def test_recompute_shadows_and_margins_replaces_stale_flags_and_syncs_status(tmp_path):
    pdf_path = str(tmp_path / "book.pdf")
    # 4 normal pages + 1 outlier large page (index 4)
    _write_synthetic_pdf(
        pdf_path,
        [
            (100, 100, 300, 500),
            (100, 100, 300, 500),
            (100, 100, 300, 500),
            (100, 100, 300, 500),
            (20, 20, 480, 680),
        ],
    )
    doc_id = "book.pdf"
    db_path = str(tmp_path / "test.db")
    store = Store(db_path)
    try:
        # Page 0: had stale MARGIN_NOT_FOUND + MARGIN_INCONSISTENT, now clean -> AUTO
        p0 = _make_page(
            pdf_path, 0, Box(100, 100, 300, 500),
            flags=[Flag.MARGIN_NOT_FOUND, Flag.MARGIN_INCONSISTENT],
            review_status=ReviewStatus.NEEDS_REVIEW,
        )
        # Page 1: had stale MARGIN_NOT_FOUND + OCR_LOW_COVERAGE -> OCR_LOW_COVERAGE kept, NEEDS_REVIEW
        p1 = _make_page(
            pdf_path, 1, Box(100, 100, 300, 500),
            flags=[Flag.MARGIN_NOT_FOUND, Flag.OCR_LOW_COVERAGE],
            review_status=ReviewStatus.NEEDS_REVIEW,
        )
        # Page 2: APPROVED with stale MARGIN_NOT_FOUND -> flag cleared, stays APPROVED
        p2 = _make_page(
            pdf_path, 2, Box(100, 100, 300, 500),
            flags=[Flag.MARGIN_NOT_FOUND],
            review_status=ReviewStatus.APPROVED,
        )
        # Page 3: clean AUTO -> stays AUTO
        p3 = _make_page(pdf_path, 3, Box(100, 100, 300, 500), review_status=ReviewStatus.AUTO)
        # Page 4: outlier large content, starts AUTO -> gets MARGIN_INCONSISTENT -> NEEDS_REVIEW
        p4 = _make_page(pdf_path, 4, Box(20, 20, 480, 680), review_status=ReviewStatus.AUTO)

        for p in (p0, p1, p2, p3, p4):
            store.upsert_page(doc_id, p.source.page_index, p)

        n = recompute_shadows_and_margins(store, doc_id, pdf_path, max_workers=2)
        assert n == 5

        by_idx = {r.page_index: r for r in store.list_pages(doc_id)}
        assert Flag.MARGIN_NOT_FOUND not in by_idx[0].params.flags
        assert Flag.MARGIN_INCONSISTENT not in by_idx[0].params.flags
        assert by_idx[0].review_status == "auto"

        assert by_idx[1].params.flags == [Flag.OCR_LOW_COVERAGE]
        assert by_idx[1].review_status == "needs_review"

        assert Flag.MARGIN_NOT_FOUND not in by_idx[2].params.flags
        assert by_idx[2].review_status == "approved"

        assert by_idx[3].params.flags == []
        assert by_idx[3].review_status == "auto"

        assert Flag.MARGIN_INCONSISTENT in by_idx[4].params.flags
        assert by_idx[4].review_status == "needs_review"
    finally:
        store.close()


def test_recompute_shadows_and_margins_drops_originals_and_matches_params_json(tmp_path, monkeypatch):
    pdf_path = str(tmp_path / "book.pdf")
    _write_synthetic_pdf(
        pdf_path,
        [
            (100, 100, 300, 500),
            (110, 105, 310, 505),
            (95, 100, 295, 495),
        ],
    )
    doc_id = "book.pdf"

    # 1) Reference: legacy behavior that holds all originals in a list
    db_ref = str(tmp_path / "ref.db")
    store_ref = Store(db_ref)
    try:
        for i in range(3):
            p = _make_page(pdf_path, i, Box(100, 100, 300, 500))
            store_ref.upsert_page(doc_id, i, p)

        from hokusai_press import nombre as nombre_mod
        from hokusai_press.source import load_single, orient

        pages_ref = [r.params for r in store_ref.list_pages(doc_id)]
        originals = []
        for p in pages_ref:
            orig, _ = load_single(pdf_path, p.source.page_index)
            cleaned, mg = pipeline_mod.compute_margin(orient(orig, p.source), p.deskew, p.regions)
            p.margin = mg
            originals.append(cleaned)
        heights = [o.shape[0] for o in originals]
        widths = [o.shape[1] for o in originals]
        nombre_mod.resolve(pages_ref, heights, widths)
        for p, flag in zip(pages_ref, margin_mod.align_margins([p.margin for p in pages_ref])):
            if flag is not None and not p.blank and flag not in p.flags:
                p.flags.append(flag)
        margin_mod.normalize_margins(pages_ref, 5.0)
        for p in pages_ref:
            store_ref.upsert_page(doc_id, p.source.page_index, p)
        ref_jsons = [
            json.dumps(asdict(r.params), default=_enc, sort_keys=True)
            for r in store_ref.list_pages(doc_id)
        ]
    finally:
        store_ref.close()

    # 2) New behavior: worker returns only (h, w, mg), and originals are freed per page
    db_new = str(tmp_path / "new.db")
    store_new = Store(db_new)
    try:
        for i in range(3):
            p = _make_page(pdf_path, i, Box(100, 100, 300, 500))
            store_new.upsert_page(doc_id, i, p)

        returned_shapes: list[tuple] = []
        orig_worker = pipeline_mod._load_and_compute_margin

        def _spy_worker(args):
            res = orig_worker(args)
            returned_shapes.append((type(res[0]), type(res[1]), type(res[2])))
            assert not any(isinstance(x, np.ndarray) for x in res)
            return res

        monkeypatch.setattr(pipeline_mod, "_load_and_compute_margin", _spy_worker)
        recompute_shadows_and_margins(store_new, doc_id, pdf_path)
        gc.collect()

        assert len(returned_shapes) == 3
        assert all(t == (int, int, Margin) for t in returned_shapes)

        new_jsons = [
            json.dumps(asdict(r.params), default=_enc, sort_keys=True)
            for r in store_new.list_pages(doc_id)
        ]
        assert new_jsons == ref_jsons
    finally:
        store_new.close()


def test_recompute_shadows_and_margins_preserves_verified_nombre(tmp_path):
    """Regression: recompute_shadows_and_margins must not clobber verified
    nombre_box / nombre_source / page_number / nombre_text with find_content_box's
    shape-only _detect_nombre heuristic or an OCR-only nombre.resolve wipe."""
    from PIL import Image

    pdf_path = str(tmp_path / "book_nombre.pdf")
    # Draw body + a decoy mark in the top band that _detect_nombre would grab
    imgs = []
    for _ in range(5):
        img = np.full((700, 500, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (100, 120), (400, 560), (0, 0, 0), -1)
        # Decoy small component in top 8% band (center) -> _detect_nombre picks this
        cv2.rectangle(img, (240, 18), (260, 34), (0, 0, 0), -1)
        imgs.append(Image.fromarray(img))
    imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:], resolution=100)

    doc_id = "book_nombre.pdf"
    db_path = str(tmp_path / "test_preserve.db")
    store = Store(db_path)
    try:
        expected = {}
        for i in range(5):
            nb = Box(380.0, 620.0, 405.0, 640.0) if i % 2 == 0 else Box(95.0, 620.0, 120.0, 640.0)
            src = "folio" if i < 3 else ("ocr" if i == 3 else None)
            p = _make_page(pdf_path, i, Box(100, 120, 400, 560), page_w=500.0, page_h=700.0)
            p.margin.nombre_box = nb
            p.margin.nombre_source = src
            p.page_number = 10 + i
            p.nombre_text = str(10 + i)
            expected[i] = (nb, src, 10 + i, str(10 + i))
            store.upsert_page(doc_id, i, p)

        # First run with align_nombre=False
        recompute_shadows_and_margins(store, doc_id, pdf_path, align_nombre=False)
        for row in store.list_pages(doc_id):
            nb, src, pnum, ntxt = expected[row.page_index]
            assert row.params.margin.nombre_box == nb
            assert row.params.margin.nombre_source == src
            assert row.params.page_number == pnum
            assert row.params.nombre_text == ntxt

        # Also run with default align_nombre=True (5 pages -> guided Pass 2a skips)
        recompute_shadows_and_margins(store, doc_id, pdf_path, align_nombre=True)
        for row in store.list_pages(doc_id):
            nb, src, pnum, ntxt = expected[row.page_index]
            assert row.params.margin.nombre_box == nb
            assert row.params.margin.nombre_source == src
            assert row.params.page_number == pnum
            assert row.params.nombre_text == ntxt
    finally:
        store.close()
