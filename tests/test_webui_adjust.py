"""Contract for WP-64a: rotate / deskew adjustment endpoints + flag info.

Backend for the review-page adjustment UI (WP-64, Antigravity):
- POST /api/page/{doc}/{i}/rotate  {"delta": 90|180|270, "decided_by": ...}
- POST /api/page/{doc}/{i}/deskew  {"angle_deg": float, "decided_by": ...}
- GET  /api/flags -> model.FLAG_INFO (for human-readable flag rendering)
Skip until implemented.
"""

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


def _seed(tmp_path, w=300, h=400):
    img_path = tmp_path / "scan.png"
    import cv2

    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (30, 40), (270, 80), (0, 0, 0), -1)
    cv2.imwrite(str(img_path), img)

    db = str(tmp_path / "t.db")
    store = Store(db)
    store.upsert_page("scan.png", 0, PageParams(
        source=SourceRef(path=str(img_path), page_index=0),
        dpi=300,
        deskew=Deskew(angle_deg=0.4, confidence=1.0),
        margin=Margin(content=Box(30, 40, 270, 80), confidence=0.7),
        review_status=ReviewStatus.NEEDS_REVIEW,
    ))
    store.close()
    return db


def _require(client):
    if client.get("/api/flags").status_code != 200:
        pytest.skip("WP-64a unimplemented (see docs/tasks/WP-64a-codex.md)")


def test_flags_endpoint_serves_flag_info(tmp_path):
    client = TestClient(create_app(_seed(tmp_path)))
    _require(client)
    info = client.get("/api/flags").json()
    assert "margin_not_found" in info
    assert set(info["margin_not_found"]) >= {"label", "desc", "check"}


def test_rotate_permutes_boxes_and_logs(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    _require(client)

    r = client.post("/api/page/scan.png/0/rotate",
                    json={"delta": 90, "decided_by": "human"})
    assert r.status_code == 200

    store = Store(db)
    try:
        p = store.get_page("scan.png", 0).params
        logged = store.decisions_for_training("rotation")
    finally:
        store.close()
    assert p.rotation == 90
    # content box (30,40,270,80) in 300x400 -> 90cw -> (400-80, 30, 400-40, 270)
    b = p.margin.content
    assert (b.x0, b.y0, b.x1, b.y1) == (320, 30, 360, 270)
    assert len(logged) == 1


def test_rotate_rejects_bad_delta(tmp_path):
    client = TestClient(create_app(_seed(tmp_path)))
    _require(client)
    r = client.post("/api/page/scan.png/0/rotate",
                    json={"delta": 45, "decided_by": "human"})
    assert r.status_code == 400


def test_rotated_output_png_has_swapped_dims(tmp_path):
    import cv2

    client = TestClient(create_app(_seed(tmp_path)))
    _require(client)
    client.post("/api/page/scan.png/0/rotate",
                json={"delta": 90, "decided_by": "human"})
    png = client.get("/img/scan.png/0/analysis.png")
    assert png.status_code == 200
    img = cv2.imdecode(np.frombuffer(png.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] > img.shape[0]  # 300x400 portrait -> landscape


def test_deskew_sets_angle_and_logs(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    _require(client)

    r = client.post("/api/page/scan.png/0/deskew",
                    json={"angle_deg": -1.2, "decided_by": "human"})
    assert r.status_code == 200

    store = Store(db)
    try:
        p = store.get_page("scan.png", 0).params
        logged = store.decisions_for_training("deskew")
    finally:
        store.close()
    assert p.deskew.angle_deg == pytest.approx(-1.2)
    assert len(logged) == 1


def test_deskew_rejects_large_angle(tmp_path):
    client = TestClient(create_app(_seed(tmp_path)))
    _require(client)
    r = client.post("/api/page/scan.png/0/deskew",
                    json={"angle_deg": 30.0, "decided_by": "human"})
    assert r.status_code == 400
