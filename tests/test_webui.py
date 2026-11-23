import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from hokusai_press.model import (
    Box,
    Deskew,
    Margin,
    PageParams,
    ReviewStatus,
    SourceRef,
)
from hokusai_press.store import Store
from hokusai_press.webui.app import create_app


def _seed(tmp_path):
    img_path = tmp_path / "scan.png"
    import cv2

    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (240, 80), (0, 0, 0), -1)
    cv2.imwrite(str(img_path), img)

    db = str(tmp_path / "t.db")
    store = Store(db)
    params = PageParams(
        source=SourceRef(path=str(img_path), page_index=0),
        dpi=300,
        deskew=Deskew(angle_deg=0.0, confidence=1.0),
        margin=Margin(content=Box(60, 60, 240, 80), confidence=0.7),
        review_status=ReviewStatus.NEEDS_REVIEW,
    )
    store.upsert_page("scan.png", 0, params)
    store.close()
    return db


def test_queue_and_decide_flow(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))

    q = client.get("/api/queue").json()
    assert q and q[0]["doc_id"] == "scan.png"

    r = client.post("/api/page/scan.png/1/decide", json={"page_kind": "bw"})
    assert r.json()["status"] == "corrected"
    # decision logged with features
    store = Store(db)
    try:
        assert len(store.decisions_for_training("page_kind")) == 1
        assert store.review_queue() == []  # cleared from queue
    finally:
        store.close()


def test_region_edit_keeps_page_in_queue_for_continuous_adds(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    # add two regions in a row
    r1 = client.post("/api/page/scan.png/1/region",
                     json={"kind": "text", "x0": 10, "y0": 10, "x1": 90, "y1": 40})
    r2 = client.post("/api/page/scan.png/1/region",
                     json={"kind": "photo", "x0": 10, "y0": 50, "x1": 90, "y1": 90})
    # page stays in the queue between adds (not finalized)
    assert r1.json()["status"] == "needs_review" and r1.json()["regions"] == 1
    assert r2.json()["status"] == "needs_review" and r2.json()["regions"] == 2
    assert client.get("/api/queue").json()  # still listed

    store = Store(db)
    try:
        regs = store.get_page("scan.png", 0).params.regions
        assert [x.source for x in regs] == ["manual", "manual"]
        assert len(store.decisions_for_training("region")) == 2
    finally:
        store.close()

    # done finalizes -> corrected, removed from queue
    fin = client.post("/api/page/scan.png/1/decide", json={"finish": True})
    assert fin.json()["status"] == "corrected"
    assert client.get("/api/queue").json() == []


def test_region_delete_removes_and_logs(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    client.post("/api/page/scan.png/1/region",
                json={"kind": "text", "x0": 0, "y0": 0, "x1": 30, "y1": 30})
    client.post("/api/page/scan.png/1/region",
                json={"kind": "photo", "x0": 40, "y0": 40, "x1": 90, "y1": 90})
    # delete the first one
    r = client.post("/api/page/scan.png/1/region/0/delete")
    assert r.json()["regions"] == 1
    store = Store(db)
    try:
        regs = store.get_page("scan.png", 0).params.regions
        assert len(regs) == 1 and regs[0].kind.value == "photo"  # the survivor
        assert len(store.decisions_for_training("region_delete")) == 1
    finally:
        store.close()


def test_set_content_updates_box_and_recomputes(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    r = client.post("/api/page/scan.png/1/content",
                    json={"x0": 20, "y0": 20, "x1": 280, "y1": 360})
    assert r.json()["recomputed"] >= 1
    store = Store(db)
    try:
        m = store.get_page("scan.png", 0).params.margin
        assert (m.content.x0, m.content.y1) == (20, 360)
        assert m.crop is not None                       # normalization ran
        assert len(store.decisions_for_training("content")) == 1
    finally:
        store.close()


def test_set_and_delete_nombre(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    client.post("/api/page/scan.png/1/nombre",
                json={"x0": 120, "y0": 360, "x1": 180, "y1": 380})
    store = Store(db)
    try:
        assert store.get_page("scan.png", 0).params.margin.nombre_box is not None
    finally:
        store.close()
    r = client.post("/api/page/scan.png/1/nombre/delete")
    assert r.json()["recomputed"] >= 1
    store = Store(db)
    try:
        assert store.get_page("scan.png", 0).params.margin.nombre_box is None
    finally:
        store.close()
    # deleting again => nothing to delete
    assert client.post("/api/page/scan.png/1/nombre/delete").status_code == 404


def test_delete_content_redetects(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    # first override content, then delete -> reverts to auto-detection
    client.post("/api/page/scan.png/1/content",
                json={"x0": 0, "y0": 0, "x1": 5, "y1": 5})
    r = client.post("/api/page/scan.png/1/content/delete")
    assert r.json()["recomputed"] >= 1
    store = Store(db)
    try:
        m = store.get_page("scan.png", 0).params.margin
        assert m.content.width > 5    # re-detected the real content bar
        assert len(store.decisions_for_training("content_delete")) == 1
    finally:
        store.close()


def test_doc_recompute_endpoint(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    assert client.post("/api/doc/scan.png/recompute").json()["recomputed"] == 1


def test_page_view_has_content_nombre_modes(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    client.post("/api/page/scan.png/1/nombre",
                json={"x0": 120, "y0": 360, "x1": 180, "y1": 380})
    html = client.get("/page/scan.png/1").text
    assert "setMode('content')" in html and "setMode('nombre')" in html
    assert "内容領域" in html and "ページ番号領域" in html
    assert "delBox('content')" in html       # content box deletable overlay
    assert "delBox('nombre')" in html        # nombre box deletable overlay
    assert "backToQueue()" in html           # back-to-queue recomputes


def test_overlays_are_grouped_for_mode_scoped_hover(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    client.post("/api/page/scan.png/1/region",
                json={"kind": "photo", "x0": 10, "y0": 10, "x1": 90, "y1": 90})
    client.post("/api/page/scan.png/1/nombre",
                json={"x0": 120, "y0": 360, "x1": 180, "y1": 380})
    html = client.get("/page/scan.png/1").text
    # each overlay tags its kind so setMode can scope hover/click to one group
    assert 'data-group="photo"' in html
    assert 'data-group="content"' in html      # seed page has a content box
    assert 'data-group="nombre"' in html
    # only the active group is interactive; others are click-through
    assert "el.dataset.group===grp" in html
    assert "el.style.pointerEvents" in html


def test_region_list_shows_classes_and_delete_buttons(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    client.post("/api/page/scan.png/1/region",
                json={"kind": "text", "x0": 0, "y0": 0, "x1": 30, "y1": 30})
    client.post("/api/page/scan.png/1/region",
                json={"kind": "photo", "x0": 40, "y0": 40, "x1": 90, "y1": 90})
    html = client.get("/page/scan.png/1").text
    assert "検出領域" in html and "text:1" in html and "photo:1" in html
    assert "delRegion(0)" in html and "delRegion(1)" in html   # per-row delete
    assert "削除" in html


def test_region_delete_out_of_range_404(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    assert client.post("/api/page/scan.png/1/region/5/delete").status_code == 404


def test_page_view_has_mode_buttons_and_dblclick_delete(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    client.post("/api/page/scan.png/1/region",
                json={"kind": "photo", "x0": 10, "y0": 10, "x1": 90, "y1": 90})
    html = client.get("/page/scan.png/1").text
    assert 'class="rbox"' in html               # clickable region overlay
    assert "ondblclick=\"delRegion(0)\"" in html  # double-click deletes, no arg event
    assert "oncontextmenu" not in html          # right-click delete removed
    assert "setMode('gray')" in html and "setMode('color')" in html  # mode buttons
    assert "addPhoto(" in html                  # drag adds immediately
    assert "id=\"mGray\"" in html and "id=\"mColor\"" in html
    # mode persists across reloads (saved + restored from localStorage)
    assert "localStorage.setItem('hp_mode'" in html
    assert "localStorage.getItem('hp_mode')" in html


def test_region_edit_photo_tone_persisted(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    r = client.post("/api/page/scan.png/1/region",
                    json={"kind": "photo", "tone": "gray",
                          "x0": 10, "y0": 10, "x1": 90, "y1": 90})
    assert r.json()["status"] == "needs_review"  # stays until finish
    store = Store(db)
    try:
        reg = store.get_page("scan.png", 0).params.regions[0]
        assert reg.kind.value == "photo" and reg.tone == "gray"
    finally:
        store.close()


def test_region_edit_tone_ignored_for_non_photo(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    client.post("/api/page/scan.png/1/region",
                json={"kind": "figure", "tone": "gray",
                      "x0": 10, "y0": 10, "x1": 90, "y1": 90})
    store = Store(db)
    try:
        reg = store.get_page("scan.png", 0).params.regions[0]
        assert reg.kind.value == "figure" and reg.tone is None
    finally:
        store.close()


def test_analysis_preview_png(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    r = client.get("/img/scan.png/1/analysis.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_previews_are_not_cached_and_versioned(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    # page embeds cache-busting tokens so a reload after an edit refetches
    html = client.get("/page/scan.png/1").text
    assert "analysis.png?v=" in html and "output.png?v=" in html
    # image responses opt out of caching
    for kind in ("analysis", "output"):
        r = client.get(f"/img/scan.png/1/{kind}.png")
        assert r.headers.get("cache-control") == "no-store"


def test_legend_colors_match_overlay():
    from hokusai_press.preview import _COLORS, _bgr_to_hex, legend
    from hokusai_press.model import RegionKind

    entries = dict(legend())
    # every region color appears in the legend with its exact hex
    for kind in (RegionKind.TEXT, RegionKind.FIGURE, RegionKind.PHOTO):
        assert _bgr_to_hex(_COLORS[kind]) in entries.values()
    # photo is red (R dominant), text is green (G dominant)
    photo_hex = _bgr_to_hex(_COLORS[RegionKind.PHOTO])
    assert photo_hex == "#dc3c3c"


def test_page_view_has_page_kind_buttons_and_drag(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    html = client.get("/page/scan.png/1").text
    # whole-page kind buttons relocated to the top, relabeled in Japanese
    assert "紙面全体が同一要素なら押す" in html
    assert "白黒二値" in html and "グレースケール" in html and "キューに戻る" in html
    assert "ボタンを押してからドラッグで領域追加" in html
    # the old legend label is gone
    assert "analysis 凡例" not in html
    # drag mapping needs the deskewed-original dims (seed image is 300x400)
    assert "ORIG_W=300" in html and "ORIG_H=400" in html


def test_pages_are_shown_by_pdf_page_number(tmp_path):
    db = _seed(tmp_path)                       # page_index 0 = PDF page 1
    client = TestClient(create_app(db))
    assert "scan.png p1</a>" in client.get("/").text
    assert "page 1</h2>" in client.get("/page/scan.png/1").text


def test_urls_name_pages_by_pdf_page_number(tmp_path):
    db = _seed(tmp_path)                       # page_index 0 = PDF page 1
    client = TestClient(create_app(db))
    assert 'href="/page/scan.png/1"' in client.get("/").text
    html = client.get("/page/scan.png/1").text
    assert "/img/scan.png/1/analysis.png" in html
    assert "/api/page/scan.png/1/decide" in html
    assert client.get("/api/queue").json()[0]["page"] == 1
    assert client.get("/api/page/scan.png/1").status_code == 200
    assert client.get("/page/scan.png/0").status_code == 404     # no page 0
    assert client.get("/api/page/scan.png/0").status_code == 404
    assert client.get("/page/scan.png/2").status_code == 404     # past the end


def test_page_view_has_approve_finish_and_back_without_recompute(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    html = client.get("/page/scan.png/1").text
    assert "承認して次へ" in html
    assert "編集を完了して次へ" in html
    assert "decide({approve:true})" in html
    assert "decide({finish:true})" in html
    # backToQueue returns to '/' without unconditionally calling /recompute
    idx_back = html.index("function backToQueue()")
    idx_decide = html.index("async function decide(")
    back_body = html[idx_back:idx_decide]
    assert "recompute" not in back_body
    assert "location.href='/'" in back_body


def test_decide_returns_next_queue_url(tmp_path):
    import cv2

    img_path = tmp_path / "book.png"
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    db = str(tmp_path / "book.db")
    store = Store(db)
    for i in range(3):
        store.upsert_page("book", i, PageParams(
            source=SourceRef(path=str(img_path), page_index=0),
            dpi=300,
            deskew=Deskew(angle_deg=0.0, confidence=1.0),
            margin=Margin(content=Box(10, 10, 190, 190), confidence=0.8),
            review_status=ReviewStatus.NEEDS_REVIEW,
        ))
    store.close()

    client = TestClient(create_app(db))
    r1 = client.post("/api/page/book/1/decide", json={"approve": True})
    assert r1.json()["status"] == "approved"
    assert r1.json()["next_url"] == "/page/book/2"

    r2 = client.post("/api/page/book/2/decide", json={"finish": True})
    assert r2.json()["status"] == "corrected"
    assert r2.json()["next_url"] == "/page/book/3"

    r3 = client.post("/api/page/book/3/decide", json={"approve": True})
    assert r3.json()["status"] == "approved"
    assert r3.json()["next_url"] == "/"


def test_delete_content_uses_compute_margin_with_regions_and_keeps_nombre(tmp_path):
    import cv2
    from hokusai_press.model import Region, RegionKind

    img_path = tmp_path / "a7.png"
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    # Small dark bar inside [40..140, 40..80] + photo content inside [160..260, 120..220]
    cv2.rectangle(img, (40, 40), (140, 80), (0, 0, 0), -1)
    cv2.rectangle(img, (170, 130), (250, 210), (120, 120, 120), -1)
    cv2.imwrite(str(img_path), img)

    db = str(tmp_path / "a7.db")
    store = Store(db)
    nb = Box(130, 260, 170, 280)
    params = PageParams(
        source=SourceRef(path=str(img_path), page_index=0),
        dpi=300,
        deskew=Deskew(angle_deg=0.0, confidence=1.0),
        margin=Margin(content=Box(0, 0, 10, 10), confidence=1.0, nombre_box=nb),
        regions=[Region(kind=RegionKind.PHOTO, box=Box(160, 120, 260, 220), source="manual")],
        review_status=ReviewStatus.NEEDS_REVIEW,
    )
    store.upsert_page("a7", 0, params)
    store.close()

    client = TestClient(create_app(db))
    r = client.post("/api/page/a7/1/content/delete")
    assert r.status_code == 200

    store = Store(db)
    try:
        m = store.get_page("a7", 0).params.margin
        # Region [160,120,260,220] must be merged into content and confidence == 1.0
        assert m.confidence == 1.0
        assert m.content.x1 >= 260 and m.content.y1 >= 220
        assert m.nombre_box == nb
    finally:
        store.close()


def test_output_png_matches_built_pdf_with_ink_valley_despeckle_and_vecfill(tmp_path):
    import cv2
    import pypdfium2 as pdfium
    from hokusai_press.model import Document, Region, RegionKind, RenderSettings
    from hokusai_press.render import build_pdf

    # Page 0: normal text page (gives dark ink_valley ~130)
    p0 = np.full((200, 200, 3), 245, dtype=np.uint8)
    cv2.rectangle(p0, (20, 20), (180, 50), (15, 15, 15), -1)
    p0_path = tmp_path / "p0.png"
    cv2.imwrite(str(p0_path), p0)

    # Page 1: show-through (light gray ~215, no dark ink) + 1px isolated speck + solid FIGURE
    p1 = np.full((200, 200, 3), 245, dtype=np.uint8)
    cv2.rectangle(p1, (15, 15), (95, 45), (215, 215, 215), -1)  # show-through -> stays white under book ink_valley
    p1[10, 10] = (0, 0, 0)                                     # 1px speck -> despeckled
    cv2.rectangle(p1, (50, 70), (150, 150), (160, 160, 160), -1)  # solid FIGURE fill
    p1_path = tmp_path / "p1.png"
    cv2.imwrite(str(p1_path), p1)

    db = str(tmp_path / "a13.db")
    store = Store(db)
    params0 = PageParams(
        source=SourceRef(path=str(p0_path), page_index=0),
        dpi=600,
        deskew=Deskew(angle_deg=0.0, confidence=1.0),
        margin=Margin(content=Box(0, 0, 200, 200), crop=Box(0, 0, 200, 200), confidence=1.0),
    )
    params1 = PageParams(
        source=SourceRef(path=str(p1_path), page_index=0),
        dpi=600,
        deskew=Deskew(angle_deg=0.0, confidence=1.0),
        margin=Margin(content=Box(0, 0, 200, 200), crop=Box(0, 0, 200, 200), confidence=1.0),
        regions=[Region(kind=RegionKind.FIGURE, box=Box(50, 70, 150, 150), source="ocr")],
        review_status=ReviewStatus.NEEDS_REVIEW,
    )
    # Seed 2 normal pages + 1 show-through page so book median ink_valley is from normal pages
    store.upsert_page("a13", 0, params0)
    store.upsert_page("a13", 1, params0)
    store.upsert_page("a13", 2, params1)
    store.close()

    client = TestClient(create_app(db))
    r = client.get("/img/a13/3/output.png")
    assert r.status_code == 200
    prev = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)

    # Build real PDF with the same document ink_valley
    from hokusai_press.geometry.margin import document_ink_valley

    valley = document_ink_valley([
        cv2.cvtColor(p0, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(p0, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY),
    ])
    doc = Document(
        source_path=str(p1_path),
        pages=[params1],
        render=RenderSettings(target_dpi=600, ink_valley=valley),
    )
    out_pdf = str(tmp_path / "out.pdf")
    build_pdf(doc, [p1], out_pdf)

    pdf = pdfium.PdfDocument(out_pdf)
    try:
        # 600 DPI page rendered at 600 DPI (scale = 600 / 72) matches 200x200 px
        bmp = pdf[0].render(scale=600 / 72).to_numpy()
    finally:
        pdf.close()

    # Check exact pixel match (or near-zero difference due to anti-aliasing on vector rect edges)
    assert prev.shape == bmp.shape
    diff = np.abs(prev.astype(np.int16) - bmp.astype(np.int16)).max(axis=2)
    diff_px = int((diff > 2).sum())
    assert diff_px == 0 or diff_px < 50
    # Specifically check speck was removed, show-through stayed white, and FIGURE center is gray ~160
    assert prev[10, 10].tolist() == [255, 255, 255]
    assert prev[30, 50].tolist() == [255, 255, 255]
    assert 140 <= int(prev[80, 60, 0]) <= 180
