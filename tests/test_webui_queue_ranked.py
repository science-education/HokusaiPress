"""Contract for WP-32: expose queue_rank ordering through the review-queue API.

docs/VIEWER_LIBRARY_PLAN.md WP-32. GET /api/queue must order each doc's pending
pages by queue_rank.rank_pending (resolves-the-most-pages first), not by raw
page_index, so the reviewer sees the highest-value page first. Skip until
implemented.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from hokusai_press.model import (
    Box,
    Deskew,
    PageParams,
    Region,
    RegionKind,
    ReviewStatus,
    SourceRef,
)
from hokusai_press.store import Store
from hokusai_press.webui.app import create_app


def _page(n_text=1, n_photo=0, angle=0.0):
    regions = [Region(kind=RegionKind.TEXT, box=Box(0, 0, 50, 10))
               for _ in range(n_text)]
    regions += [Region(kind=RegionKind.PHOTO, box=Box(0, 20, 80, 90))
                for _ in range(n_photo)]
    return PageParams(
        source=SourceRef(path="s.pdf", page_index=0),
        deskew=Deskew(angle_deg=angle, confidence=1.0),
        regions=regions,
        review_status=ReviewStatus.NEEDS_REVIEW,
    )


def _seed(tmp_path):
    db = str(tmp_path / "t.db")
    store = Store(db)
    # A 3-page cluster (indices 5,6,7) plus one isolated outlier (index 1),
    # inserted so raw page_index order would put the outlier first.
    store.upsert_page("doc", 1, _page(n_text=9, n_photo=0, angle=9.0))
    store.upsert_page("doc", 5, _page(n_photo=4))
    store.upsert_page("doc", 6, _page(n_photo=4))
    store.upsert_page("doc", 7, _page(n_photo=4))
    store.close()
    return db


def _require(client):
    r = client.get("/api/queue")
    body = r.json()
    if not body or "rank_score" not in body[0]:
        pytest.skip("WP-32 unimplemented (see docs/tasks/WP-32-codex.md)")
    return body


def test_queue_orders_clustered_page_before_outlier(tmp_path):
    client = TestClient(create_app(_seed(tmp_path)))
    rows = _require(client)

    idxs = [r["page_index"] for r in rows if r["doc_id"] == "doc"]
    assert set(idxs) == {1, 5, 6, 7}
    # A cluster member (highest resolves-count) must come before the outlier.
    assert idxs.index(1) == len(idxs) - 1


def test_queue_still_filters_to_needs_review(tmp_path):
    db = _seed(tmp_path)
    store = Store(db)
    store.upsert_page("doc", 9, PageParams(
        source=SourceRef(path="s.pdf", page_index=9),
        review_status=ReviewStatus.AUTO,
    ))
    store.close()

    client = TestClient(create_app(db))
    rows = _require(client)
    assert 9 not in [r["page_index"] for r in rows]
