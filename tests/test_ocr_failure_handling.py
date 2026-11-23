import cv2
import numpy as np
import pytest

from hokusai_press import pipeline
from hokusai_press.model import Flag, ReviewStatus
from hokusai_press.store import Store


def _write_sparse_ink_page(path, black_w: int = 11, black_h: int = 4) -> str:
    """Create a 400x300 white page with 44 black pixels (4x11) inside."""
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    img[150:150 + black_h, 120:120 + black_w] = 0
    cv2.imwrite(str(path), img)
    return str(path)


def _render_pdf_page_gray(pdf_path: str, page_index: int = 0) -> np.ndarray:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    try:
        bitmap = doc[page_index].render(scale=6.0)
        arr = bitmap.to_numpy()
    finally:
        doc.close()
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def test_run_aborts_on_ocr_import_error_without_creating_outputs(tmp_path, monkeypatch):
    src = _write_sparse_ink_page(tmp_path / "sparse.png")
    out_pdf = tmp_path / "out.pdf"
    db_path = tmp_path / "hokusai.db"

    def missing_engine(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'missing_ocr_pkg'", name="missing_ocr_pkg")

    import hokusai_press.ocr

    monkeypatch.setattr(hokusai_press.ocr, "get_ocr_engine", missing_engine)

    with pytest.raises(ImportError) as exc_info:
        pipeline.run(src, str(out_pdf), db_path=str(db_path), use_ocr=True)

    msg = str(exc_info.value)
    assert "missing_ocr_pkg" in msg
    assert "--no-ocr" in msg
    assert not out_pdf.exists()
    assert not db_path.exists()


def test_per_page_ocr_exception_preserves_ink_and_flags_review(tmp_path, monkeypatch):
    pytest.importorskip("pikepdf")
    pytest.importorskip("pypdfium2")

    src = _write_sparse_ink_page(tmp_path / "sparse.png")
    out_pdf = tmp_path / "out.pdf"
    db_path = tmp_path / "hokusai.db"

    class FailingEngine:
        def __call__(self, image_bgr):
            raise RuntimeError("synthetic per-page OCR failure")

    import hokusai_press.ocr

    monkeypatch.setattr(hokusai_press.ocr, "get_ocr_engine", lambda *a, **kw: FailingEngine())

    summary = pipeline.run(src, str(out_pdf), db_path=str(db_path), use_ocr=True)

    assert summary["needs_review"] == [0]
    assert out_pdf.exists()

    store = Store(str(db_path))
    try:
        row = store.list_pages("sparse.png")[0]
    finally:
        store.close()

    assert row.params.blank is False
    assert Flag.OCR_FAILED in row.params.flags
    assert row.params.review_status == ReviewStatus.NEEDS_REVIEW

    rendered = _render_pdf_page_gray(str(out_pdf))
    assert int((rendered < 128).sum()) > 0


def test_no_ocr_preserves_sparse_text(tmp_path):
    pytest.importorskip("pikepdf")
    pytest.importorskip("pypdfium2")

    src = _write_sparse_ink_page(tmp_path / "sparse.png")
    out_pdf = tmp_path / "out.pdf"
    db_path = tmp_path / "hokusai.db"

    pipeline.run(src, str(out_pdf), db_path=str(db_path), use_ocr=False)

    assert out_pdf.exists()
    store = Store(str(db_path))
    try:
        row = store.list_pages("sparse.png")[0]
    finally:
        store.close()

    assert row.params.blank is False
    rendered = _render_pdf_page_gray(str(out_pdf))
    assert int((rendered < 128).sum()) > 0


def test_ocr_success_no_regions_blank_page_becomes_white(tmp_path, monkeypatch):
    pytest.importorskip("pikepdf")
    pytest.importorskip("pypdfium2")

    src = _write_sparse_ink_page(tmp_path / "blankish.png")
    out_pdf = tmp_path / "out.pdf"
    db_path = tmp_path / "hokusai.db"

    class EmptyOCREngine:
        def __call__(self, image_bgr):
            return {"lines": [], "layout_boxes": []}

    import hokusai_press.ocr

    monkeypatch.setattr(hokusai_press.ocr, "get_ocr_engine", lambda *a, **kw: EmptyOCREngine())

    pipeline.run(src, str(out_pdf), db_path=str(db_path), use_ocr=True)

    assert out_pdf.exists()
    store = Store(str(db_path))
    try:
        row = store.list_pages("blankish.png")[0]
    finally:
        store.close()

    assert row.params.blank is True
    assert Flag.OCR_FAILED not in row.params.flags
    assert Flag.NO_TEXT not in row.params.flags

    rendered = _render_pdf_page_gray(str(out_pdf))
    assert int((rendered < 250).sum()) == 0
