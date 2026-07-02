"""Contract for WP-11a (backend of reading mode) — docs/VIEWER_LIBRARY_PLAN.md.

Written BEFORE implementation. Defines the API the reading UI (WP-11/12,
Antigravity) consumes. Codex implements the endpoints to make this green.
Do not weaken the assertions.

Two endpoints:
- GET  /api/doc/{doc_id}/pages       ordered list of ALL pages (reading grid),
                                      each with needs_review + optional label.
- POST /api/page/{doc_id}/{page_index}/report   one-tap "this page looks wrong"
                                      from the reader: flags the page for review
                                      and logs it as a decision (teacher signal).
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


def _seed(tmp_path):
    img_path = tmp_path / "scan.png"
    import cv2

    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (240, 80), (0, 0, 0), -1)
    cv2.imwrite(str(img_path), img)

    db = str(tmp_path / "t.db")
    store = Store(db)

    def _p(idx, status):
        return PageParams(
            source=SourceRef(path=str(img_path), page_index=0),
            dpi=300,
            deskew=Deskew(angle_deg=0.0, confidence=1.0),
            margin=Margin(content=Box(60, 60, 240, 80), confidence=0.7),
            review_status=status,
        )

    # Two AUTO pages and one already flagged, out of insertion order.
    store.upsert_page("scan.png", 2, _p(2, ReviewStatus.AUTO))
    store.upsert_page("scan.png", 0, _p(0, ReviewStatus.AUTO))
    store.upsert_page("scan.png", 1, _p(1, ReviewStatus.NEEDS_REVIEW))
    store.close()
    return db


def _require_impl(db):
    """Pending contract: skip (not fail) until WP-11a lands, so the committed
    suite stays green. When Codex adds GET /api/doc/{id}/pages the probe returns
    200 and every test below activates automatically."""
    client = TestClient(create_app(db))
    if client.get("/api/doc/scan.png/pages").status_code != 200:
        pytest.skip(
            "WP-11a unimplemented: reading-mode endpoints missing "
            "(see docs/tasks/WP-11a-codex.md)"
        )
    return client


def test_pages_listing_is_ordered_and_complete(tmp_path):
    client = _require_impl(_seed(tmp_path))

    rows = client.get("/api/doc/scan.png/pages").json()

    # ALL pages (not just the review queue), in reading order.
    assert [r["page_index"] for r in rows] == [0, 1, 2]
    # Each row tells the reader whether the page is flagged.
    flagged = {r["page_index"]: r["needs_review"] for r in rows}
    assert flagged == {0: False, 1: True, 2: False}


def test_report_flags_page_and_logs_decision(tmp_path):
    db = _seed(tmp_path)
    client = _require_impl(db)

    r = client.post("/api/page/scan.png/0/report", json={"decided_by": "human"})
    assert r.status_code == 200
    assert r.json()["status"] == "flagged"

    # The page is now in the review queue.
    q = client.get("/api/queue").json()
    assert any(p["doc_id"] == "scan.png" and p["page_index"] == 0 for p in q)

    # And the report was logged as a teacher signal for learning.
    store = Store(db)
    try:
        logged = store.decisions_for_training("reader_report")
    finally:
        store.close()
    assert len(logged) == 1


def test_report_unknown_page_is_404(tmp_path):
    client = _require_impl(_seed(tmp_path))
    r = client.post("/api/page/scan.png/99/report", json={"decided_by": "human"})
    assert r.status_code == 404
