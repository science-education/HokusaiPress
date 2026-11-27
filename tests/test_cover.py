"""Tests for cover fetching, upscaling, and cover+blank-verso PDF insertion."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pikepdf
import pytest
from PIL import Image

from hokusai_press import cli, cover
from hokusai_press.model import Box, PageParams, Region, RegionKind, SourceRef
from hokusai_press.store import Store


def _make_jpeg_bytes(w: int, h: int, color: tuple[int, int, int] = (200, 80, 40)) -> bytes:
    img = Image.new("RGB", (w, h), color)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_body_pdf(path: Path, n_pages: int = 3, w_pt: float = 420.0, h_pt: float = 595.0,
                   text_pages: dict[int, str] | None = None) -> None:
    """Create a minimal multi-page PDF with optional text via reportlab."""
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(w_pt, h_pt))
    for i in range(n_pages):
        if text_pages and i in text_pages:
            c.drawString(40, h_pt - 60, text_pages[i])
        c.showPage()
    c.save()
    path.write_bytes(buf.getvalue())


class _FakeResp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_isbn10_to_13():
    assert cover.isbn10_to_13("4-00-000000-4") == "9784000000000"
    assert cover.isbn10_to_13("4873115655") == "9784873115658"
    assert cover.isbn10_to_13("978-4-87311-565-8") == "9784873115658"
    with pytest.raises(ValueError):
        cover.isbn10_to_13("12345")


def test_fetch_cover_openbd_and_cache(tmp_path: Path):
    jpg = _make_jpeg_bytes(400, 600)
    calls: list[str] = []

    def fake_open(req, timeout=20):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        calls.append(url)
        if "api.openbd.jp" in url:
            body = json.dumps([{"summary": {"cover": "https://cover.openbd.jp/9784873115658.jpg"}}]).encode()
            return _FakeResp(body)
        if "cover.openbd.jp" in url:
            return _FakeResp(jpg)
        raise AssertionError(f"unexpected url: {url}")

    cache_dir = tmp_path / "cache"
    res = cover.fetch_cover_image("4873115655", cache_dir=str(cache_dir), opener=fake_open)
    assert res is not None
    assert res.isbn == "9784873115658"
    assert res.source == "openbd"
    assert not res.cached
    assert res.image_bgr.shape == (600, 400, 3)

    # Second call hits disk cache without HTTP
    calls.clear()
    res2 = cover.fetch_cover_image("9784873115658", cache_dir=str(cache_dir), opener=fake_open)
    assert res2 is not None
    assert res2.cached
    assert res2.source == "cache"
    assert calls == []


def test_fetch_cover_hanmoto_and_google_fallback():
    jpg = _make_jpeg_bytes(350, 500)

    def fake_open_hanmoto(req, timeout=20):
        url = req.full_url
        if "api.openbd.jp" in url:
            return _FakeResp(b"[null]")
        if "www.hanmoto.com/bd/isbn/" in url:
            html_body = '<meta property="og:image" content="https://www.hanmoto.com/bd/img/9784873115658.jpg">'
            return _FakeResp(html_body.encode())
        if "hanmoto.com/bd/img/" in url:
            return _FakeResp(jpg)
        raise AssertionError(f"unexpected url: {url}")

    res = cover.fetch_cover_image("9784873115658", opener=fake_open_hanmoto)
    assert res is not None
    assert res.source == "hanmoto"

    def fake_open_google(req, timeout=20):
        url = req.full_url
        if "api.openbd.jp" in url:
            return _FakeResp(b"[null]")
        if "hanmoto.com" in url:
            return _FakeResp(b"<html>no image</html>")
        if "googleapis.com/books" in url:
            payload = {"items": [{"volumeInfo": {"imageLinks": {"thumbnail": "http://books.google.com/c.jpg&edge=curl"}}}]}
            return _FakeResp(json.dumps(payload).encode())
        if "books.google.com/c.jpg" in url:
            return _FakeResp(jpg)
        raise AssertionError(f"unexpected url: {url}")

    res_g = cover.fetch_cover_image("9784873115658", opener=fake_open_google)
    assert res_g is not None
    assert res_g.source == "google"


def test_fetch_cover_skips_tiny_thumbnail():
    tiny = _make_jpeg_bytes(80, 120)
    good = _make_jpeg_bytes(400, 600)

    def fake_open(req, timeout=20):
        url = req.full_url
        if "api.openbd.jp" in url:
            return _FakeResp(json.dumps([{"summary": {"cover": "https://cover.openbd.jp/tiny.jpg"}}]).encode())
        if "tiny.jpg" in url:
            return _FakeResp(tiny)
        if "www.hanmoto.com/bd/isbn/" in url:
            return _FakeResp(b'<meta property="og:image" content="https://www.hanmoto.com/bd/img/good.jpg">')
        if "good.jpg" in url:
            return _FakeResp(good)
        raise AssertionError(url)

    res = cover.fetch_cover_image("9784873115658", opener=fake_open)
    assert res is not None
    assert res.source == "hanmoto"


def test_upscale_cover_lanczos_and_skip():
    small = np.full((600, 400, 3), 128, dtype=np.uint8)
    up, method = cover.upscale_cover(small, target_long_edge=2400, mode="auto")
    assert method == "lanczos+usm"
    assert up.shape[:2] == (2400, 1600)

    large = np.full((2200, 1500, 3), 128, dtype=np.uint8)
    same, method2 = cover.upscale_cover(large, target_long_edge=2400, mode="auto")
    assert method2 == "original"
    assert same.shape == large.shape


def test_insert_cover_pages_no_margins_and_blank_verso(tmp_path: Path):
    src = tmp_path / "book.pdf"
    dst = tmp_path / "book_cover.pdf"
    _make_body_pdf(src, n_pages=3, w_pt=420.0, h_pt=595.0)

    # Cover image has 2:3 aspect ratio (400x600), different from A5 (420x595)
    cover_bgr = np.full((600, 400, 3), 90, dtype=np.uint8)
    res = cover.insert_cover_pages(
        str(src), str(dst), cover_bgr,
        isbn="9784873115658", source_name="openbd", target_dpi=300,
    )
    assert not res.skipped
    assert res.upscale_method == "lanczos+usm"

    with pikepdf.open(dst) as pdf:
        assert len(pdf.pages) == 5  # cover + blank + 3 body
        assert str(pdf.docinfo[cover.COVER_META_KEY]) == "9784873115658"

        # Page 0 (cover): height matches body (595.0), width preserves 2:3 ratio (no margins!)
        mb0 = [float(v) for v in pdf.pages[0].mediabox]
        assert mb0[3] == pytest.approx(595.0, abs=0.1)
        assert mb0[2] == pytest.approx(595.0 * (400.0 / 600.0), abs=0.2)

        # Page 1 (blank verso): matches body page size
        mb1 = [float(v) for v in pdf.pages[1].mediabox]
        assert mb1[2] == pytest.approx(420.0, abs=0.1)
        assert mb1[3] == pytest.approx(595.0, abs=0.1)

    # Idempotency: second run skips unless force/replace_cover
    dst2 = tmp_path / "book_cover2.pdf"
    res2 = cover.insert_cover_pages(str(dst), str(dst2), cover_bgr, isbn="9784873115658")
    assert res2.skipped
    assert not dst2.exists()

    # replace_cover replaces page 0 (cover) and adds blank verso -> 6 pages if blank_back=True,
    # or 5 pages if no_blank_back
    res3 = cover.insert_cover_pages(
        str(dst), str(dst2), cover_bgr,
        isbn="9784873115658", blank_back=False, replace_cover=True,
    )
    assert not res3.skipped
    with pikepdf.open(dst2) as pdf2:
        assert len(pdf2.pages) == 5


def test_extract_isbn_from_pdf_and_db_fallback(tmp_path: Path):
    # 1) PDF with text layer
    pdf_txt = tmp_path / "with_text.pdf"
    _make_body_pdf(pdf_txt, n_pages=2, text_pages={1: "ISBN 978-4-87311-565-8"})
    isbn, origin = cover.extract_isbn(str(pdf_txt), lookup=None)
    assert isbn == "9784873115658"
    assert origin == "pdf_text"

    # 2) Image-only PDF (no text layer) -> fallback to Store DB
    pdf_img = tmp_path / "scan_01_HP.pdf"
    _make_body_pdf(pdf_img, n_pages=2, text_pages=None)

    db_path = tmp_path / "hokusai.db"
    store = Store(str(db_path))
    try:
        p0 = PageParams(source=SourceRef(path="scan_01.pdf", page_index=0),
                        regions=[Region(kind=RegionKind.TEXT, box=Box(0, 0, 10, 10), ocr_text="Title")])
        p1 = PageParams(source=SourceRef(path="scan_01.pdf", page_index=1),
                        regions=[Region(kind=RegionKind.TEXT, box=Box(0, 0, 10, 10),
                                        ocr_text="2026年 発行 ISBN 4-87311-565-5")])
        store.replace_doc_pages("scan_01.pdf", [p0, p1])
    finally:
        store.close()

    isbn2, origin2 = cover.extract_isbn(str(pdf_img), db_path=str(db_path), lookup=None)
    assert isbn2 == "9784873115658"
    assert origin2 == "db"


def test_cli_cover_local_image_and_dry_run(tmp_path: Path):
    src = tmp_path / "book.pdf"
    _make_body_pdf(src, n_pages=2)
    img_path = tmp_path / "cover.jpg"
    img_path.write_bytes(_make_jpeg_bytes(400, 600))
    out_pdf = tmp_path / "out.pdf"

    rc_dry = cli.main([
        "cover", str(src), "--out", str(out_pdf),
        "--cover-image", str(img_path), "--dry-run",
    ])
    assert rc_dry == 0
    assert not out_pdf.exists()

    rc = cli.main([
        "cover", str(src), "--out", str(out_pdf),
        "--cover-image", str(img_path), "--isbn", "9784873115658",
    ])
    assert rc == 0
    assert out_pdf.is_file()
    with pikepdf.open(out_pdf) as pdf:
        assert len(pdf.pages) == 4
