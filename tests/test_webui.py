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

    r = client.post("/api/page/scan.png/0/decide", json={"page_kind": "bw"})
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
    r1 = client.post("/api/page/scan.png/0/region",
                     json={"kind": "text", "x0": 10, "y0": 10, "x1": 90, "y1": 40})
    r2 = client.post("/api/page/scan.png/0/region",
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
    fin = client.post("/api/page/scan.png/0/decide", json={"finish": True})
    assert fin.json()["status"] == "corrected"
    assert client.get("/api/queue").json() == []


def test_region_delete_removes_and_logs(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    client.post("/api/page/scan.png/0/region",
                json={"kind": "text", "x0": 0, "y0": 0, "x1": 30, "y1": 30})
    client.post("/api/page/scan.png/0/region",
                json={"kind": "photo", "x0": 40, "y0": 40, "x1": 90, "y1": 90})
    # delete the first one
    r = client.post("/api/page/scan.png/0/region/0/delete")
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
    r = client.post("/api/page/scan.png/0/content",
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
    client.post("/api/page/scan.png/0/nombre",
                json={"x0": 120, "y0": 360, "x1": 180, "y1": 380})
    store = Store(db)
    try:
        assert store.get_page("scan.png", 0).params.margin.nombre_box is not None
    finally:
        store.close()
    r = client.post("/api/page/scan.png/0/nombre/delete")
    assert r.json()["recomputed"] >= 1
    store = Store(db)
    try:
        assert store.get_page("scan.png", 0).params.margin.nombre_box is None
    finally:
        store.close()
    # deleting again => nothing to delete
    assert client.post("/api/page/scan.png/0/nombre/delete").status_code == 404


def test_delete_content_redetects(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    # first override content, then delete -> reverts to auto-detection
    client.post("/api/page/scan.png/0/content",
                json={"x0": 0, "y0": 0, "x1": 5, "y1": 5})
    r = client.post("/api/page/scan.png/0/content/delete")
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
    client.post("/api/page/scan.png/0/nombre",
                json={"x0": 120, "y0": 360, "x1": 180, "y1": 380})
    html = client.get("/page/scan.png/0").text
    assert "setMode('content')" in html and "setMode('nombre')" in html
    assert "内容領域" in html and "ページ番号領域" in html
    assert "delBox('content')" in html       # content box deletable overlay
    assert "delBox('nombre')" in html        # nombre box deletable overlay
    assert "backToQueue()" in html           # back-to-queue recomputes


def test_region_delete_out_of_range_404(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    assert client.post("/api/page/scan.png/0/region/5/delete").status_code == 404


def test_page_view_has_mode_buttons_and_dblclick_delete(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    client.post("/api/page/scan.png/0/region",
                json={"kind": "photo", "x0": 10, "y0": 10, "x1": 90, "y1": 90})
    html = client.get("/page/scan.png/0").text
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
    r = client.post("/api/page/scan.png/0/region",
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
    client.post("/api/page/scan.png/0/region",
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
    r = client.get("/img/scan.png/0/analysis.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_previews_are_not_cached_and_versioned(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    # page embeds cache-busting tokens so a reload after an edit refetches
    html = client.get("/page/scan.png/0").text
    assert "analysis.png?v=" in html and "output.png?v=" in html
    # image responses opt out of caching
    for kind in ("analysis", "output"):
        r = client.get(f"/img/scan.png/0/{kind}.png")
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
    html = client.get("/page/scan.png/0").text
    # whole-page kind buttons relocated to the top, relabeled in Japanese
    assert "紙面全体が同一要素なら押す" in html
    assert "白黒二値" in html and "グレースケール" in html and "キューに戻る" in html
    assert "ボタンを押してからドラッグで領域追加" in html
    # the old legend label is gone
    assert "analysis 凡例" not in html
    # drag mapping needs the deskewed-original dims (seed image is 300x400)
    assert "ORIG_W=300" in html and "ORIG_H=400" in html
