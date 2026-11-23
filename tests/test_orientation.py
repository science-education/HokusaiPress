"""Scanner-bed cropping and content-based page orientation."""

import cv2
import numpy as np

from hokusai_press import orientation
from hokusai_press.model import SourceRef, _decode_page
from hokusai_press.source import find_paper, orient


def _bed_scan(page_gray=70, w=3510, h=2235, x0=1030, x1=2480):
    """A black page scanned with the gray bed around it (img20261121_0010 p15)."""
    rng = np.random.default_rng(0)
    img = np.full((h, w), 123, np.float32) + rng.normal(0, 0.8, (h, w))
    img[:, x0:x1] = page_gray + rng.normal(0, 4, (h, x1 - x0))
    img[:, :31] = 255
    img[:, -31:] = 255
    img[:31, :] = 255
    img[-31:, :] = 255
    cv2.putText(img, "TITLE", (1400, 600), cv2.FONT_HERSHEY_SIMPLEX, 3, 240, 6)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_find_paper_crops_the_scanner_bed():
    box = find_paper(_bed_scan())
    assert box is not None
    x0, y0, x1, y1 = box
    assert abs(x0 - 1030) < 15 and abs(x1 - 2480) < 15
    assert x1 - x0 < y1 - y0                 # the sheet is portrait


def test_find_paper_leaves_an_ordinary_page_alone():
    img = np.full((2209, 1448), 250, np.uint8)
    img[:, :31] = 255
    img[:, -31:] = 255
    cv2.putText(img, "text", (300, 600), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 6)
    assert find_paper(img) is None


def test_orient_crops_then_turns_clockwise():
    img = np.zeros((100, 200), np.uint8)
    img[10, 50] = 255
    out = orient(img, SourceRef(path="x", paper=[40, 0, 120, 100], rotate=90))
    assert out.shape == (80, 100)
    ys, xs = np.nonzero(out)
    assert (ys[0], xs[0]) == (10, 89)        # (x=10,y=10) in the crop -> clockwise


def test_choose_rotation_keeps_the_reading_that_looks_like_text():
    class Engine:
        def __call__(self, img):
            # pretend the page reads properly only when it is portrait
            if img.shape[0] > img.shape[1]:
                return {"lines": [{"text": "これは本文のひらがなです"}]}
            return {"lines": [{"text": "罒丨亅乚"}]}

    portrait_when = {90: (300, 200), 270: (300, 200)}
    img = np.zeros((200, 300), np.uint8)     # landscape: both turns give portrait
    r, scores = orientation.choose_rotation(img, Engine())
    assert r in portrait_when and scores[r][1] > 0

    class Upright:
        def __call__(self, img):
            # text reads only when the original top edge (marked) is on the left
            return {"lines": [{"text": "ひらがな" if img[0, 0] == 255 else "罒"}]}

    img = np.zeros((200, 300), np.uint8)
    img[-1, 0] = 255                          # bottom-left -> top-left after 90 CW
    r, _ = orientation.choose_rotation(img, Upright())
    assert r == 90


def test_old_params_with_the_reoriented_flag_turn_90():
    d = {"source": {"path": "x", "page_index": 3},
         "flags": ["page_reoriented_to_match_book"]}
    assert _decode_page(d).source.rotate == 90
    d = {"source": {"path": "x", "page_index": 3, "rotate": 0, "paper": [1, 2, 3, 4]},
         "flags": ["page_reoriented_to_match_book"]}
    p = _decode_page(d)
    assert p.source.rotate == 0 and p.source.paper == [1, 2, 3, 4]


def test_layout_score_tells_upright_from_upside_down_tategaki():
    # columns 40 wide, right to left; full columns run 100..1700, a paragraph
    # ends short (100..900), the next starts indented one character (140..1700)
    cols = []
    for k, (y0, y1) in enumerate([(100, 1700), (100, 900), (140, 1700), (100, 1700),
                                  (100, 1300), (140, 1700), (100, 1700), (100, 600)]):
        x1 = 1200 - 76 * k
        cols.append({"box": (x1 - 40, y0, x1, y1)})
    up = orientation.layout_score(cols)
    flipped = [{"box": (1400 - x1, 1800 - y1, 1400 - x0, 1800 - y0)}
               for x0, y0, x1, y1 in (c["box"] for c in cols)]
    assert up > 0 > orientation.layout_score(flipped)


def _bars_img(h=380, w=240, span=170):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    x1 = min(w - 16, 16 + span)
    for y in range(24, h - 24, 18):
        img[y:y + 4, 16:x1] = 0
    return img


def _make_pdf_from_arrays(path, arrays, dpi=300.0):
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    c = Canvas(str(path))
    for arr in arrays:
        h, w = arr.shape[:2]
        pw, ph = w * 72.0 / dpi, h * 72.0 / dpi
        c.setPageSize((pw, ph))
        rgb = arr[:, :, ::-1] if arr.ndim == 3 else arr
        c.drawImage(ImageReader(Image.fromarray(rgb)), 0, 0, pw, ph)
        c.showPage()
    c.save()
    return str(path)


def test_a8_partial_pages_crops_bed_and_rotates_via_pdf_sizes_and_db(tmp_path, monkeypatch):
    """A8: 1-2 page --pages runs must still crop the scanner bed and unify
    orientation: (1) when DB has no full book, using pypdfium2 get_size() across
    the PDF; (2) when DB already has the full book, inheriting SourceRef
    (paper, rotate) from the stored page."""
    import hokusai_press.source as source_mod
    from hokusai_press import pipeline
    from hokusai_press.store import Store

    # Page 4 (index 3): landscape scan with gray bed around a portrait dark sheet
    bed = np.full((300, 480, 3), 125, dtype=np.uint8)
    bed[:, 125:355] = 65
    bed[100:115, 172:270] = 245
    bed[:4] = bed[-4:] = 255
    bed[:, :4] = bed[:, -4:] = 255

    # 5-page PDF: 3 portrait pages + page 4 (bed) + page 5 (landscape rotated)
    pdf_path = _make_pdf_from_arrays(
        tmp_path / "orientation.pdf",
        [_bars_img(380, 240)] * 3 + [bed, _bars_img(240, 360)],
    )

    # Case 1: Fresh DB (no prior full run) with 1 or 2 pages in --pages
    db_fresh = str(tmp_path / "fresh.db")
    pipeline.run(
        pdf_path, str(tmp_path / "fresh_sub.pdf"),
        db_path=db_fresh, use_ocr=False, pages={3, 4},
    )
    st = Store(db_fresh)
    try:
        p3 = st.get_page("orientation.pdf", 3).params
        p4 = st.get_page("orientation.pdf", 4).params
        assert p3.source.paper == [125.0, 0.0, 355.0, 300.0]
        assert p3.source.rotate == 0
        assert p4.source.paper is None
        assert p4.source.rotate == 90
    finally:
        st.close()

    # Case 2: Full run first, then partial --pages 4-5 and --pages 5 on same DB
    db_full = str(tmp_path / "full.db")
    pipeline.run(pdf_path, str(tmp_path / "full.pdf"), db_path=db_full, use_ocr=False)

    find_calls = []
    orig_find = source_mod.find_paper
    monkeypatch.setattr(
        source_mod, "find_paper",
        lambda img: (find_calls.append(1), orig_find(img))[1],
    )
    pipeline.run(
        pdf_path, str(tmp_path / "sub45.pdf"),
        db_path=db_full, use_ocr=False, pages={3, 4},
    )
    pipeline.run(
        pdf_path, str(tmp_path / "sub5.pdf"),
        db_path=db_full, use_ocr=False, pages={4},
    )
    # Inherited from stored full-book DB without re-running find_paper
    assert find_calls == []
    st = Store(db_full)
    try:
        p3 = st.get_page("orientation.pdf", 3).params
        p4 = st.get_page("orientation.pdf", 4).params
        assert p3.source.paper == [125.0, 0.0, 355.0, 300.0]
        assert p3.source.rotate == 0
        assert p4.source.paper is None
        assert p4.source.rotate == 90
    finally:
        st.close()

