import os

from hokusai_press.cli import (
    _expand_sources,
    _is_folder_out,
    _parse_pages,
    _resolve_out,
)


def test_remargin_skips_ocr(tmp_path, monkeypatch):
    """remargin must rebuild from stored regions without calling OCR again --
    that's the whole point (margin/figure tweaks are cheap to re-verify)."""
    import cv2
    import numpy as np
    import pytest

    pytest.importorskip("hybrid_ocr")  # pipeline.run -> build_pdf needs it
    pytest.importorskip("pikepdf")

    from hokusai_press import pipeline
    from hokusai_press.cli import main

    img = np.full((1000, 700, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 120), (600, 880), (0, 0, 0), 2)
    for y in range(160, 840, 40):
        cv2.rectangle(img, (120, y), (580, y + 16), (0, 0, 0), -1)
    src = str(tmp_path / "pg.png")
    cv2.imwrite(src, img)

    db = str(tmp_path / "j.db")
    pipeline.run(src, str(tmp_path / "o.pdf"), db_path=db, use_ocr=False)

    calls = []
    import hokusai_press.content as content_mod
    monkeypatch.setattr(content_mod, "analyze",
                        lambda *a, **k: calls.append(1) or (None, None))

    out = str(tmp_path / "remargin.pdf")
    rc = main(["remargin", src, "--out", out, "--db", db])
    assert rc == 0
    assert os.path.exists(out)
    assert calls == []  # content.analyze (OCR) was never invoked


def test_parse_pages():
    assert _parse_pages(None) is None
    assert _parse_pages("") is None
    assert _parse_pages("9") == {9}
    assert _parse_pages("52,236-237") == {52, 236, 237}
    assert _parse_pages("0-4") == {0, 1, 2, 3, 4}
    assert _parse_pages(" 3 , 5 ") == {3, 5}


def test_out_file_path_is_used_verbatim(tmp_path):
    out = str(tmp_path / "book.pdf")
    assert _resolve_out(out, "C:/scans/chufu.pdf") == out


def test_existing_dir_uses_source_stem(tmp_path):
    out = _resolve_out(str(tmp_path), "C:/scans/chufu.pdf")
    assert out == os.path.join(str(tmp_path), "chufu.pdf")


def test_trailing_separator_is_a_folder(tmp_path):
    folder = str(tmp_path / "out") + os.sep
    out = _resolve_out(folder, "/data/shinto.pdf")
    assert out == os.path.join(folder, "shinto.pdf")
    assert os.path.isdir(folder)            # created on demand


def test_extensionless_is_treated_as_folder(tmp_path):
    folder = str(tmp_path / "results")
    out = _resolve_out(folder, "img20260420_0057.pdf")
    assert out == os.path.join(folder, "img20260420_0057.pdf")
    assert os.path.isdir(folder)


def test_image_source_becomes_pdf(tmp_path):
    out = _resolve_out(str(tmp_path), "/data/page01.png")
    assert out == os.path.join(str(tmp_path), "page01.pdf")


def _touch(p):
    with open(p, "wb") as f:
        f.write(b"%PDF-1.4\n")
    return str(p)


def test_expand_directory_to_pdfs(tmp_path):
    a = _touch(tmp_path / "b.pdf")
    b = _touch(tmp_path / "a.pdf")
    _touch(tmp_path / "note.txt")           # ignored
    got = _expand_sources([str(tmp_path)])
    assert [os.path.basename(p) for p in got] == ["a.pdf", "b.pdf"]  # sorted


def test_expand_glob(tmp_path):
    _touch(tmp_path / "scan1.pdf")
    _touch(tmp_path / "scan2.pdf")
    _touch(tmp_path / "other.pdf")
    got = _expand_sources([os.path.join(str(tmp_path), "scan*.pdf")])
    assert sorted(os.path.basename(p) for p in got) == ["scan1.pdf", "scan2.pdf"]


def test_expand_plain_file_and_dedupe(tmp_path):
    f = _touch(tmp_path / "x.pdf")
    # same file via explicit path and via directory expansion -> de-duplicated
    got = _expand_sources([f, str(tmp_path)])
    assert got == [f]


def test_is_folder_out():
    assert _is_folder_out("C:/out") is True          # extensionless
    assert _is_folder_out("C:/out/") is True          # trailing sep
    assert _is_folder_out("C:/out/book.pdf") is False  # explicit file


def test_merge_dbs_folds_workers_and_cleans_up(tmp_path):
    from hokusai_press.cli import _merge_dbs
    from hokusai_press.model import Box, Margin, PageParams, SourceRef
    from hokusai_press.store import Store

    def _seed(path, doc, pages):
        s = Store(path)
        for i in range(pages):
            s.upsert_page(doc, i, PageParams(
                source=SourceRef(path=doc, page_index=i),
                margin=Margin(content=Box(0, 0, 10, 10))))
        s.close()

    w0 = str(tmp_path / "w0.db")
    w1 = str(tmp_path / "w1.db")
    _seed(w0, "a.pdf", 3)
    _seed(w1, "b.pdf", 2)
    target = str(tmp_path / "merged.db")
    _merge_dbs(target, [w0, w1])

    tgt = Store(target)
    try:
        assert set(tgt.doc_ids()) == {"a.pdf", "b.pdf"}
        assert len(tgt.list_pages("a.pdf")) == 3
        assert len(tgt.list_pages("b.pdf")) == 2
    finally:
        tgt.close()
    assert not os.path.exists(w0) and not os.path.exists(w1)   # temp dbs removed
