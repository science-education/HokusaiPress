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


def test_analysis_preview_png(tmp_path):
    db = _seed(tmp_path)
    client = TestClient(create_app(db))
    r = client.get("/img/scan.png/0/analysis.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
