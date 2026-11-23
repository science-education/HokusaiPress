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


def _make_npage_pdf(path, n_pages: int) -> str:
    import cv2
    import numpy as np
    from PIL import Image

    imgs = []
    for i in range(n_pages):
        arr = np.full((300, 200, 3), 255, np.uint8)
        cv2.rectangle(arr, (20, 20), (180, 280), (0, 0, 0), 2)
        cv2.putText(arr, f"P{i + 1}", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        imgs.append(Image.fromarray(arr))
    p = str(path)
    imgs[0].save(p, save_all=True, append_images=imgs[1:], resolution=100)
    return p


def test_doc_conflict_different_folders_both_exist(tmp_path, capsys):
    """A/scan.pdf -> B/scan.pdf (both exist): 2nd run and remargin fail with exit code 1,
    and DB stays pointing to A."""
    pytest.importorskip("pikepdf")
    from hokusai_press.cli import main
    from hokusai_press.model import same_path
    from hokusai_press.store import Store

    dir_a = tmp_path / "A"
    dir_b = tmp_path / "B"
    out_a = tmp_path / "outA"
    out_b = tmp_path / "outB"
    dir_a.mkdir()
    dir_b.mkdir()

    pdf_a = _make_npage_pdf(dir_a / "scan.pdf", 3)
    pdf_b = _make_npage_pdf(dir_b / "scan.pdf", 2)
    db = str(tmp_path / "hokusai.db")

    rc1 = main(["run", pdf_a, "--out", str(out_a), "--db", db, "--no-ocr"])
    assert rc1 == 0
    capsys.readouterr()

    rc2 = main(["run", pdf_b, "--out", str(out_b), "--db", db, "--no-ocr"])
    out2 = capsys.readouterr().out
    assert rc2 == 1
    assert pdf_a in out2 and pdf_b in out2
    assert "--db" in out2

    rc_rem = main(["remargin", pdf_b, "--out", str(out_b), "--db", db])
    out_rem = capsys.readouterr().out
    assert rc_rem == 1
    assert pdf_a in out_rem and pdf_b in out_rem

    store = Store(db)
    try:
        rows = store.list_pages("scan.pdf")
        assert len(rows) == 3
        assert all(same_path(r.params.source.path, pdf_a) for r in rows)
    finally:
        store.close()


def test_doc_moved_after_original_deleted_replaces_pages(tmp_path):
    """After processing A/scan.pdf (3 pages), delete A and run B/scan.pdf (2 pages):
    succeeds, all stored pages come from B, and count matches B (no trailing old rows)."""
    pytest.importorskip("pikepdf")
    from hokusai_press.cli import main
    from hokusai_press.model import same_path
    from hokusai_press.store import Store

    dir_a = tmp_path / "A"
    dir_b = tmp_path / "B"
    out_dir = tmp_path / "out"
    dir_a.mkdir()
    dir_b.mkdir()

    pdf_a = _make_npage_pdf(dir_a / "scan.pdf", 3)
    pdf_b = _make_npage_pdf(dir_b / "scan.pdf", 2)
    db = str(tmp_path / "hokusai.db")

    assert main(["run", pdf_a, "--out", str(out_dir), "--db", db, "--no-ocr"]) == 0
    os.remove(pdf_a)

    assert main(["run", pdf_b, "--out", str(out_dir), "--db", db, "--no-ocr"]) == 0

    store = Store(db)
    try:
        rows = store.list_pages("scan.pdf")
        assert [r.page_index for r in rows] == [0, 1]
        assert all(same_path(r.params.source.path, pdf_b) for r in rows)
        assert len(store.page_features("scan.pdf")) == 2
    finally:
        store.close()

    # Rebuild on B succeeds without trying to read non-existent page 3
    rebuilt = str(tmp_path / "rebuilt_b.pdf")
    assert main(["rebuild", pdf_b, "--doc", "scan.pdf", "--out", rebuilt, "--db", db]) == 0
    assert os.path.exists(rebuilt)


def test_same_path_replaced_with_shorter_pdf_prunes_trailing_pages(tmp_path):
    """Replacing the same source path from 3 pages to 1 page and running full-page analysis
    leaves exactly 1 row in DB."""
    pytest.importorskip("pikepdf")
    from hokusai_press.cli import main
    from hokusai_press.store import Store

    src = tmp_path / "scan.pdf"
    out_dir = tmp_path / "out"
    db = str(tmp_path / "hokusai.db")

    _make_npage_pdf(src, 3)
    assert main(["run", str(src), "--out", str(out_dir), "--db", db, "--no-ocr"]) == 0

    _make_npage_pdf(src, 1)
    assert main(["run", str(src), "--out", str(out_dir), "--db", db, "--no-ocr"]) == 0

    store = Store(db)
    try:
        rows = store.list_pages("scan.pdf")
        assert [r.page_index for r in rows] == [0]
        assert len(store.page_features("scan.pdf")) == 1
    finally:
        store.close()


def test_partial_pages_run_preserves_other_pages(tmp_path):
    """--pages partial processing must NOT delete other pages in DB."""
    pytest.importorskip("pikepdf")
    from hokusai_press.cli import main
    from hokusai_press.store import Store

    src = _make_npage_pdf(tmp_path / "scan.pdf", 3)
    out_dir = tmp_path / "out"
    db = str(tmp_path / "hokusai.db")

    assert main(["run", src, "--out", str(out_dir), "--db", db, "--no-ocr"]) == 0
    assert main(["run", src, "--out", str(out_dir), "--db", db, "--no-ocr", "--pages", "1"]) == 0

    store = Store(db)
    try:
        rows = store.list_pages("scan.pdf")
        assert [r.page_index for r in rows] == [0, 1, 2]
        assert len(store.page_features("scan.pdf")) == 3
    finally:
        store.close()


def test_single_run_duplicate_basenames_fails(tmp_path, capsys):
    """A single run with dir1/x.pdf and dir2/x.pdf fails before processing."""
    from hokusai_press.cli import main
    from hokusai_press.store import Store

    d1 = tmp_path / "dir1"
    d2 = tmp_path / "dir2"
    out_dir = tmp_path / "out"
    d1.mkdir()
    d2.mkdir()
    p1 = _make_npage_pdf(d1 / "x.pdf", 1)
    p2 = _make_npage_pdf(d2 / "x.pdf", 1)
    db = str(tmp_path / "hokusai.db")

    # Even if output collision were avoided or checked, duplicate doc_id must fail cleanly.
    rc = main(["run", p1, p2, "--out", str(out_dir), "--db", db, "--no-ocr"])
    out_text = capsys.readouterr().out
    assert rc == 1
    assert p1 in out_text and p2 in out_text
    if os.path.exists(db):
        s = Store(db)
        try:
            assert s.doc_ids() == []
        finally:
            s.close()


def test_rebuild_with_fewer_pdf_pages_than_db_reports_clean_error(tmp_path, capsys):
    """If DB has 3 pages but the PDF on disk only has 1 page, rebuild prints a clear error
    and exits with code 1 instead of crashing with a raw traceback."""
    pytest.importorskip("pikepdf")
    from hokusai_press.cli import main
    from hokusai_press.model import Box, Deskew, Margin, PageParams, SourceRef
    from hokusai_press.store import Store

    src = _make_npage_pdf(tmp_path / "short.pdf", 1)
    db = str(tmp_path / "hokusai.db")
    store = Store(db)
    for i in range(3):
        store.upsert_page("short.pdf", i, PageParams(
            source=SourceRef(path=src, page_index=i),
            deskew=Deskew(),
            margin=Margin(content=Box(20, 20, 180, 280), confidence=1.0),
        ))
    store.close()

    out_pdf = str(tmp_path / "rebuilt.pdf")
    rc = main(["rebuild", src, "--doc", "short.pdf", "--out", out_pdf, "--db", db])
    out_text = capsys.readouterr().out
    assert rc == 1
    assert "error:" in out_text
    assert "short.pdf" in out_text


def _dump_db_tables(db_path: str) -> dict:
    """Read all relevant tables from *db_path*, stripping timestamp columns so
    serial and parallel runs can be compared for exact content equality."""
    from hokusai_press.store import Store

    s = Store(db_path)
    try:
        books = [
            {k: v for k, v in dict(r).items() if k != "resolved_at"}
            for r in s.conn.execute("SELECT * FROM book ORDER BY book_id").fetchall()
        ]
        scan_profiles = [
            {k: v for k, v in dict(r).items() if k != "created_at"}
            for r in s.conn.execute("SELECT * FROM scan_profile ORDER BY book_id").fetchall()
        ]
        page_features = [
            dict(r)
            for r in s.conn.execute(
                "SELECT * FROM page_feature ORDER BY book_id, page_index"
            ).fetchall()
        ]
        region_features = [
            dict(r)
            for r in s.conn.execute(
                "SELECT * FROM region_feature ORDER BY book_id, page_index, cls, x0, y0, x1, y1"
            ).fetchall()
        ]
        pages = [
            {k: v for k, v in dict(r).items() if k != "updated_at"}
            for r in s.conn.execute(
                "SELECT * FROM pages ORDER BY doc_id, page_index"
            ).fetchall()
        ]
        return {
            "book": books,
            "scan_profile": scan_profiles,
            "page_feature": page_features,
            "region_feature": region_features,
            "pages": pages,
        }
    finally:
        s.close()


def test_workers_merges_all_tables_matching_serial(tmp_path):
    """Serial (--workers 1) and parallel (--workers 2) runs on the same inputs
    must produce identical book, scan_profile, page_feature, region_feature, and
    pages rows (excluding timestamps), and prune stale trailing rows on full runs."""
    pytest.importorskip("pikepdf")
    from hokusai_press.cli import main
    from hokusai_press.model import Box, Deskew, Margin, PageParams, SourceRef
    from hokusai_press.store import Store

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pdf_a = _make_npage_pdf(src_dir / "a.pdf", 2)
    pdf_b = _make_npage_pdf(src_dir / "b.pdf", 3)

    out_serial = tmp_path / "out_serial"
    out_parallel = tmp_path / "out_parallel"
    db_serial = str(tmp_path / "serial.db")
    db_parallel = str(tmp_path / "parallel.db")

    # Seed a stale page_index=5 row in both DBs for a.pdf to verify replace_doc_pages pruning
    for db_p in (db_serial, db_parallel):
        st = Store(db_p)
        st.upsert_page(
            "a.pdf",
            5,
            PageParams(
                source=SourceRef(path=pdf_a, page_index=5),
                deskew=Deskew(),
                margin=Margin(content=Box(0, 0, 10, 10)),
            ),
        )
        st.close()

    rc_s = main(["run", pdf_a, pdf_b, "--out", str(out_serial), "--db", db_serial, "--no-ocr", "--workers", "1"])
    rc_p = main(["run", pdf_a, pdf_b, "--out", str(out_parallel), "--db", db_parallel, "--no-ocr", "--workers", "2"])
    assert rc_s == 0
    assert rc_p == 0

    dump_s = _dump_db_tables(db_serial)
    dump_p = _dump_db_tables(db_parallel)

    assert len(dump_p["book"]) == 2
    assert len(dump_p["scan_profile"]) == 2
    assert len(dump_p["page_feature"]) == 5  # 2 + 3 (stale page 5 pruned)
    assert len(dump_p["pages"]) == 5
    assert dump_p == dump_s


def test_workers_partial_pages_upserts_only_without_overwriting_profile(tmp_path):
    """With --pages on --workers 2, only the requested pages are upserted and
    existing whole-book profile tables and other pages remain intact."""
    pytest.importorskip("pikepdf")
    from hokusai_press.cli import main
    from hokusai_press.store import Store

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pdf_a = _make_npage_pdf(src_dir / "a.pdf", 3)
    pdf_b = _make_npage_pdf(src_dir / "b.pdf", 3)
    out_dir = tmp_path / "out"
    db = str(tmp_path / "hokusai.db")

    assert main(["run", pdf_a, pdf_b, "--out", str(out_dir), "--db", db, "--no-ocr", "--workers", "2"]) == 0
    before = _dump_db_tables(db)
    assert len(before["pages"]) == 6
    assert len(before["page_feature"]) == 6

    # Partial run of page 1 only with --workers 2
    assert main(["run", pdf_a, pdf_b, "--out", str(out_dir), "--db", db, "--no-ocr", "--workers", "2", "--pages", "1"]) == 0
    after = _dump_db_tables(db)
    assert len(after["pages"]) == 6
    assert after["book"] == before["book"]
    assert after["scan_profile"] == before["scan_profile"]
    assert after["page_feature"] == before["page_feature"]

    st = Store(db)
    try:
        assert [r.page_index for r in st.list_pages("a.pdf")] == [0, 1, 2]
        assert [r.page_index for r in st.list_pages("b.pdf")] == [0, 1, 2]
    finally:
        st.close()


def test_workers_one_book_fails_other_succeeds_and_merges(tmp_path, capsys, monkeypatch):
    """When 1 of 2 books raises an exception in a worker, the successful book is
    still merged into the target DB, [FAIL] is printed for the broken book, and
    main returns exit code 1."""
    pytest.importorskip("pikepdf")
    import hokusai_press.pipeline as pipeline_mod
    from hokusai_press.cli import main
    from hokusai_press.store import Store

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pdf_ok = _make_npage_pdf(src_dir / "good.pdf", 2)
    pdf_bad = _make_npage_pdf(src_dir / "bad.pdf", 2)
    out_dir = tmp_path / "out"
    db = str(tmp_path / "hokusai.db")

    orig_run = pipeline_mod.run

    def _flaky_run(path, *args, **kwargs):
        if os.path.basename(path) == "bad.pdf":
            raise RuntimeError("corrupt scan simulation")
        return orig_run(path, *args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "run", _flaky_run)

    rc = main(["run", pdf_ok, pdf_bad, "--out", str(out_dir), "--db", db, "--no-ocr", "--workers", "2"])
    out_text = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] bad.pdf: corrupt scan simulation" in out_text
    assert "[OK] good.pdf:" in out_text

    st = Store(db)
    try:
        assert st.doc_ids() == ["good.pdf"]
        assert len(st.list_pages("good.pdf")) == 2
        assert len(st.page_features("good.pdf")) == 2
        book_row = st.conn.execute("SELECT * FROM book WHERE book_id='good.pdf'").fetchone()
        assert book_row is not None
        sp_row = st.conn.execute("SELECT * FROM scan_profile WHERE book_id='good.pdf'").fetchone()
        assert sp_row is not None
    finally:
        st.close()


def test_workers_keeps_temp_db_on_merge_failure(tmp_path, capsys, monkeypatch):
    """If merging a worker's temp DB fails, the temp DB is preserved on disk and
    its path is reported in the [FAIL] output."""
    pytest.importorskip("pikepdf")
    from hokusai_press.cli import main
    from hokusai_press.store import Store

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pdf_a = _make_npage_pdf(src_dir / "a.pdf", 1)
    pdf_b = _make_npage_pdf(src_dir / "b.pdf", 1)
    out_dir = tmp_path / "out"
    db = str(tmp_path / "hokusai.db")

    orig_merge = Store.merge_from

    def _failing_merge(self, other, full_run=True):
        if "b.pdf" in other.doc_ids():
            raise RuntimeError("disk full during merge")
        return orig_merge(self, other, full_run=full_run)

    monkeypatch.setattr(Store, "merge_from", _failing_merge)

    rc = main(["run", pdf_a, pdf_b, "--out", str(out_dir), "--db", db, "--no-ocr", "--workers", "2"])
    out_text = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] b.pdf:" in out_text
    assert "disk full during merge" in out_text

    # Extract kept temp db path from output and verify it still exists while a.pdf's was removed
    import re
    m = re.search(r"temp DB kept at (\S+\.db)", out_text)
    assert m is not None, out_text
    kept_wdb = m.group(1)
    assert os.path.exists(kept_wdb)

    # Clean up kept temp db + dir
    os.remove(kept_wdb)
    try:
        os.rmdir(os.path.dirname(kept_wdb))
    except OSError:
        pass


def test_skip_existing_checks_out_pdf_image_only_and_db(tmp_path, capsys):
    """--skip-existing only skips when primary PDF exists, requested
    --image-only-out exists, and the target DB has pages for doc_id."""
    pytest.importorskip("pikepdf")
    from hokusai_press.cli import main
    from hokusai_press.store import Store

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    pdf_a = _make_npage_pdf(src_dir / "a.pdf", 2)
    out_dir = tmp_path / "out"
    img_dir = tmp_path / "img"
    db = str(tmp_path / "hokusai.db")

    # 1. Full run creates out/a.pdf, img/a.pdf, and DB entries
    assert main([
        "run", pdf_a, "--out", str(out_dir), "--image-only-out", str(img_dir),
        "--db", db, "--no-ocr",
    ]) == 0
    capsys.readouterr()

    # All three present -> skipped
    assert main([
        "run", pdf_a, "--out", str(out_dir), "--image-only-out", str(img_dir),
        "--db", db, "--no-ocr", "--skip-existing",
    ]) == 0
    out_text = capsys.readouterr().out
    assert "[skip] a.pdf" in out_text

    # 2. Remove DB pages for a.pdf (main PDF & image-only PDF still exist) -> must NOT skip
    st = Store(db)
    st.conn.execute("DELETE FROM pages WHERE doc_id='a.pdf'")
    st.conn.commit()
    st.close()

    assert main([
        "run", pdf_a, "--out", str(out_dir), "--image-only-out", str(img_dir),
        "--db", db, "--no-ocr", "--skip-existing",
    ]) == 0
    out_text = capsys.readouterr().out
    assert "[skip] a.pdf" not in out_text
    assert "[OK] a.pdf:" in out_text

    st = Store(db)
    try:
        assert len(st.list_pages("a.pdf")) == 2
    finally:
        st.close()

    # 3. Remove image-only PDF (main PDF & DB pages still exist) -> must NOT skip when --image-only-out is requested
    img_pdf = img_dir / "a.pdf"
    os.remove(img_pdf)
    assert not img_pdf.exists()

    assert main([
        "run", pdf_a, "--out", str(out_dir), "--image-only-out", str(img_dir),
        "--db", db, "--no-ocr", "--skip-existing",
    ]) == 0
    out_text = capsys.readouterr().out
    assert "[skip] a.pdf" not in out_text
    assert "[OK] a.pdf:" in out_text
    assert img_pdf.exists()


def test_output_parent_dirs_created_before_save(tmp_path):
    """A10: _resolve_out, identify --csv, learn --out, and image_only_copy create
    missing parent directories before writing."""
    from hokusai_press import learn as learn_mod
    from hokusai_press.cli import _resolve_out, main
    from hokusai_press.model import (
        Box, Deskew, Margin, PageKind, PageParams, Region, RegionKind, SourceRef,
    )
    from hokusai_press.store import Store

    # 1. _resolve_out with a file path (has .pdf extension) creates parent dir
    nested_pdf = tmp_path / "missing_a" / "nested_b" / "book.pdf"
    resolved = _resolve_out(str(nested_pdf), "in.pdf")
    assert resolved == str(nested_pdf)
    assert nested_pdf.parent.is_dir()

    # 2. identify --csv creates missing parent dir
    src_pdf = _make_npage_pdf(tmp_path / "src.pdf", 1)
    csv_out = tmp_path / "csv_dir" / "sub" / "proposals.csv"
    assert main(["identify", src_pdf, "--csv", str(csv_out)]) == 0
    assert csv_out.is_file()

    # 3. learn --out creates missing parent dir
    db = str(tmp_path / "learn.db")
    st = Store(db)
    try:
        for i in range(6):
            kind = PageKind.BW if i < 3 else PageKind.GRAY
            p = PageParams(
                source=SourceRef(path=src_pdf, page_index=i),
                dpi=300.0,
                deskew=Deskew(0.0, 1.0),
                margin=Margin(content=Box(10, 10, 90, 90), confidence=1.0),
                regions=[Region(RegionKind.TEXT, Box(10, 10, 90, 90))],
                page_kind=kind,
            )
            st.upsert_page("src.pdf", i, p)
            st.log_decision(
                "src.pdf", i, "human", "page_kind", "auto", kind.value,
                learn_mod.page_features(p),
            )
    finally:
        st.close()
    model_out = tmp_path / "model_dir" / "sub" / "model.json"
    assert main(["learn", "--db", db, "--out", str(model_out)]) == 0
    assert model_out.is_file()
    assert learn_mod.load(str(model_out)) is not None


