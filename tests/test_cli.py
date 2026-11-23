import os

import pytest

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


def test_rebuild_pages_filter_limits_output_without_touching_stored_data(tmp_path):
    """rebuild(pages=...) renders only a subset (a fast preview) but must not
    drop or alter the OTHER stored pages -- a second rebuild with no filter
    must still produce the full page count."""
    import pytest

    pytest.importorskip("hybrid_ocr")
    pytest.importorskip("pikepdf")

    import cv2
    import numpy as np
    import pypdfium2  # noqa: F401 -- just confirms it's importable here

    from hokusai_press import pipeline
    from hokusai_press.model import Box, Deskew, Margin, PageParams, SourceRef
    from hokusai_press.store import Store

    # a tiny multi-page PDF via img2pdf-less direct PIL save (no OCR needed --
    # rebuild only reads stored params, not regions detection).
    from PIL import Image

    pages_img = []
    for i in range(5):
        img = np.full((400, 300, 3), 255, np.uint8)
        cv2.rectangle(img, (40, 40), (260, 360), (0, 0, 0), 2)
        pages_img.append(Image.fromarray(img))
    src = str(tmp_path / "multi.pdf")
    pages_img[0].save(src, save_all=True, append_images=pages_img[1:])

    db = str(tmp_path / "j.db")
    store = Store(db)
    doc_id = "multi.pdf"
    for i in range(5):
        store.upsert_page(doc_id, i, PageParams(
            source=SourceRef(path=src, page_index=i),
            deskew=Deskew(),
            margin=Margin(content=Box(40, 40, 260, 360), confidence=1.0),
        ))
    store.close()

    out_subset = str(tmp_path / "subset.pdf")
    summary = pipeline.rebuild(doc_id, src, out_subset, db_path=db, pages={1, 2})
    assert summary["pages"] == 2

    out_full = str(tmp_path / "full.pdf")
    summary_full = pipeline.rebuild(doc_id, src, out_full, db_path=db)
    assert summary_full["pages"] == 5


def test_parse_pages():
    # PDF page numbers in (1 = the first page), 0-based page indices out
    assert _parse_pages(None) is None
    assert _parse_pages("") is None
    assert _parse_pages("10") == {9}
    assert _parse_pages("53,237-238") == {52, 236, 237}
    assert _parse_pages("1-5") == {0, 1, 2, 3, 4}
    assert _parse_pages(" 4 , 6 ") == {3, 5}
    for bad in ("0", "0-4", "5-3", "x", "3-"):
        with pytest.raises(ValueError):
            _parse_pages(bad)


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


def test_rebuild_turns_pages_that_analyze_reoriented(tmp_path):
    """A landscape page that analyze turned 90 deg CW to match a portrait book
    has all its stored boxes in the turned frame, so rebuild must turn it
    too (it rendered sideways with a misplaced text layer before)."""
    import pytest

    pytest.importorskip("pikepdf")
    import numpy as np
    import pypdfium2 as pdfium
    from PIL import Image

    from hokusai_press import pipeline
    from hokusai_press.model import Deskew, Flag, PageParams, SourceRef
    from hokusai_press.store import Store

    src = str(tmp_path / "land.pdf")
    Image.fromarray(np.full((300, 400, 3), 255, np.uint8)).save(src, resolution=100)
    db = str(tmp_path / "j.db")
    store = Store(db)
    store.upsert_page("land.pdf", 0, PageParams(
        source=SourceRef(path=src, page_index=0, rotate=90), dpi=100, deskew=Deskew(),
        flags=[Flag.PAGE_REORIENTED]))
    store.close()

    out = str(tmp_path / "out.pdf")
    pipeline.rebuild("land.pdf", src, out, db_path=db, margin_mode="none")
    w, h = pdfium.PdfDocument(out)[0].get_size()
    assert h > w


def _make_sample_pdf(path) -> tuple[str, bytes]:
    import numpy as np
    from PIL import Image

    p = str(path)
    Image.fromarray(np.full((200, 150, 3), 255, np.uint8)).save(p, resolution=100)
    with open(p, "rb") as f:
        return p, f.read()


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def test_refuse_overwrite_direct_file(tmp_path, capsys, monkeypatch):
    from hokusai_press.cli import main

    src, orig_bytes = _make_sample_pdf(tmp_path / "scan.pdf")

    called = []
    import hokusai_press.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "run", lambda *a, **k: called.append("run"))
    monkeypatch.setattr(pipeline_mod, "rebuild", lambda *a, **k: called.append("rebuild"))
    monkeypatch.setattr(
        pipeline_mod, "recompute_shadows_and_margins", lambda *a, **k: called.append("remargin")
    )

    for argv in (
        ["run", src, "--out", src, "--no-ocr"],
        ["rebuild", src, "--doc", "scan.pdf", "--out", src],
        ["remargin", src, "--out", src],
        ["ocr", src, "--out", src],
    ):
        rc = main(argv)
        out_text = capsys.readouterr().out
        assert rc == 1
        assert _read_bytes(src) == orig_bytes
        assert "scan.pdf" in out_text
        assert "--suffix" in out_text
    assert called == []


def test_refuse_overwrite_source_folder(tmp_path, capsys):
    from hokusai_press.cli import main

    src_dir = tmp_path / "books"
    src_dir.mkdir()
    src, orig_bytes = _make_sample_pdf(src_dir / "scan.pdf")

    for argv in (
        ["run", src, "--out", str(src_dir), "--no-ocr"],
        ["rebuild", src, "--doc", "scan.pdf", "--out", str(src_dir)],
        ["remargin", src, "--out", str(src_dir)],
        ["ocr", src, "--out", str(src_dir)],
    ):
        rc = main(argv)
        out_text = capsys.readouterr().out
        assert rc == 1
        assert _read_bytes(src) == orig_bytes
        assert "scan.pdf" in out_text
        assert "--suffix" in out_text


def test_refuse_overwrite_case_and_relative_path(tmp_path, capsys, monkeypatch):
    from hokusai_press.cli import main

    sub = tmp_path / "sub"
    sub.mkdir()
    src, orig_bytes = _make_sample_pdf(sub / "Scan.pdf")

    monkeypatch.chdir(tmp_path)
    # Relative path with parent traversal and different casing / forward slashes
    rel_out = "./sub/../SUB/scan.pdf"
    rc = main(["run", src, "--out", rel_out, "--no-ocr"])
    out_text = capsys.readouterr().out
    assert rc == 1
    assert _read_bytes(src) == orig_bytes
    assert rel_out in out_text
    assert src in out_text
    assert "--suffix" in out_text


def test_refuse_multiple_inputs_output_collision(tmp_path, capsys):
    from hokusai_press.cli import main

    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    out_dir = tmp_path / "out"
    d1.mkdir()
    d2.mkdir()
    src1, bytes1 = _make_sample_pdf(d1 / "book.pdf")
    src2, bytes2 = _make_sample_pdf(d2 / "book.pdf")

    for argv in (
        ["run", src1, src2, "--out", str(out_dir), "--no-ocr"],
        ["remargin", src1, src2, "--out", str(out_dir)],
        ["ocr", src1, src2, "--out", str(out_dir)],
    ):
        rc = main(argv)
        out_text = capsys.readouterr().out
        assert rc == 1
        assert _read_bytes(src1) == bytes1
        assert _read_bytes(src2) == bytes2
        assert src1 in out_text
        assert src2 in out_text
        assert "--suffix" in out_text


def test_refuse_image_only_out_collision(tmp_path, capsys):
    from hokusai_press.cli import main

    src, orig_bytes = _make_sample_pdf(tmp_path / "scan.pdf")
    out_pdf = str(tmp_path / "out.pdf")

    # --image-only-out hits source
    rc1 = main(["run", src, "--out", out_pdf, "--image-only-out", src, "--no-ocr"])
    capsys.readouterr()
    assert rc1 == 1
    assert _read_bytes(src) == orig_bytes
    assert not os.path.exists(out_pdf)

    # --image-only-out hits primary --out
    rc2 = main(["rebuild", src, "--doc", "scan.pdf", "--out", out_pdf, "--image-only-out", out_pdf])
    capsys.readouterr()
    assert rc2 == 1
    assert _read_bytes(src) == orig_bytes


def test_allow_suffix_and_different_folder(tmp_path):
    pytest.importorskip("hybrid_ocr")
    pytest.importorskip("pikepdf")

    from hokusai_press.cli import main

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src, orig_bytes = _make_sample_pdf(src_dir / "scan.pdf")
    db = str(tmp_path / "j.db")

    # Same folder with --suffix succeeds and preserves source
    rc1 = main(["run", src, "--out", str(src_dir), "--suffix", "_HP", "--db", db, "--no-ocr"])
    assert rc1 == 0
    assert os.path.exists(src_dir / "scan_HP.pdf")
    assert _read_bytes(src) == orig_bytes

    # Different folder without --suffix succeeds and preserves source
    out_dir = tmp_path / "out"
    rc2 = main(["run", src, "--out", str(out_dir), "--db", db, "--no-ocr"])
    assert rc2 == 0
    assert os.path.exists(out_dir / "scan.pdf")
    assert _read_bytes(src) == orig_bytes


def test_save_safety_net_refuses_overwrite(tmp_path):
    import numpy as np
    from hokusai_press import pipeline
    from hokusai_press.mrc import MrcPageBuilder
    from hokusai_press.textlayer import image_only_copy

    src, orig_bytes = _make_sample_pdf(tmp_path / "scan.pdf")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        pipeline.run(src, src, db_path=str(tmp_path / "j.db"), use_ocr=False)
    assert _read_bytes(src) == orig_bytes

    with pytest.raises(ValueError, match="refusing to overwrite"):
        pipeline.rebuild("scan.pdf", src, src, db_path=str(tmp_path / "j.db"))
    assert _read_bytes(src) == orig_bytes

    with pytest.raises(ValueError, match="refusing to overwrite"):
        image_only_copy(src, src)
    assert _read_bytes(src) == orig_bytes

    builder = MrcPageBuilder()
    builder.add_page(np.full((100, 100, 3), 255, np.uint8), [], [], "bw")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        builder.save(src, source_path=src)
    assert _read_bytes(src) == orig_bytes
