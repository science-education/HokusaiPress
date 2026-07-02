"""Contract for WP-13a: library + search HTTP endpoints (docs/VIEWER_LIBRARY_PLAN.md).

Backs the library screen (WP-13, Antigravity). Exposes the WP-01 full-text
search and a per-doc listing over HTTP. Skip until implemented.
"""

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from hokusai_press.model import (
    Box,
    PageParams,
    Region,
    RegionKind,
    SourceRef,
)
from hokusai_press.store import Store
from hokusai_press.webui.app import create_app


def _seed(tmp_path):
    db = str(tmp_path / "t.db")
    store = Store(db)

    def _p(text, idx):
        return PageParams(
            source=SourceRef(path="scan.pdf", page_index=idx),
            regions=[Region(kind=RegionKind.TEXT, box=Box(0, 0, 100, 100),
                            ocr_text=text)],
        )

    store.upsert_page("bookA.pdf", 0, _p("量子力学の基礎", 0))
    store.upsert_page("bookA.pdf", 1, _p("相対性理論の入門", 1))
    store.upsert_page("bookB.pdf", 0, _p("古典文学の世界", 0))
    store.save_book("bookA.pdf", title="物理学入門", author="朝永")
    store.close()
    return db


def _require(db):
    client = TestClient(create_app(db))
    if client.get("/api/library").status_code != 200:
        pytest.skip("WP-13a unimplemented (see docs/tasks/WP-13a-codex.md)")
    return client


def test_library_lists_docs_with_counts(tmp_path):
    client = _require(_seed(tmp_path))
    rows = client.get("/api/library").json()

    by_doc = {r["doc_id"]: r for r in rows}
    assert set(by_doc) == {"bookA.pdf", "bookB.pdf"}
    assert by_doc["bookA.pdf"]["page_count"] == 2
    assert by_doc["bookB.pdf"]["page_count"] == 1
    # Title comes from the book table when present, else None/absent is fine.
    assert by_doc["bookA.pdf"].get("title") == "物理学入門"


def test_search_endpoint_returns_hits(tmp_path):
    client = _require(_seed(tmp_path))
    hits = client.get("/api/search", params={"q": "量子力学"}).json()

    assert [(h["doc_id"], h["page_index"]) for h in hits] == [("bookA.pdf", 0)]
    assert "量子力学" in hits[0]["snippet"]


def test_search_endpoint_empty_query_is_empty(tmp_path):
    client = _require(_seed(tmp_path))
    assert client.get("/api/search", params={"q": ""}).json() == []
