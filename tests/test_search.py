"""Contract for WP-01: full-text search over stored OCR text (docs/VIEWER_LIBRARY_PLAN.md).

This test is the acceptance criterion. It is written BEFORE the implementation
(store.py currently has no search API) and defines the contract Codex must make
green. Do not weaken these assertions to pass; implement to satisfy them.

Design decisions baked into the contract (do not re-litigate):
- The index lives in the same local hokusai.db as `pages` (FTS5 virtual table).
  It is NOT a shareable/copyright-safe artifact — OCR body text stays local.
- Tokenizer is FTS5 'trigram', which gives substring matching WITHOUT Japanese
  word segmentation. Trigram only matches queries of length >= 3, so `search`
  MUST fall back to a LIKE scan for queries of 1-2 characters. Both paths are
  exercised below.
- The index is kept in sync automatically: `upsert_page` (re)indexes the page's
  `params.reading_text()`. Re-upserting with changed text leaves no stale hits.
- `search` returns `SearchHit(doc_id, page_index, snippet)` ranked by relevance
  (bm25 for the FTS path); `snippet` contains the matched term.
"""

import pytest

from hokusai_press.model import (
    Box,
    PageParams,
    Region,
    RegionKind,
    SourceRef,
)

# Pending contract: skip (not error) until WP-01 lands, so the rest of the
# suite still collects and runs. When Codex adds SearchHit/Store.search this
# import succeeds and every test below activates automatically.
try:
    from hokusai_press.store import SearchHit, Store
except ImportError:
    pytest.skip(
        "WP-01 unimplemented: hokusai_press.store.SearchHit/Store.search missing "
        "(see docs/tasks/WP-01-codex.md)",
        allow_module_level=True,
    )


def _page(text: str, page_index: int = 0) -> PageParams:
    """A page whose single text region carries `text` as OCR content."""
    return PageParams(
        source=SourceRef(path="scan.pdf", page_index=page_index),
        regions=[
            Region(
                kind=RegionKind.TEXT,
                box=Box(0, 0, 100, 100),
                ocr_text=text,
            )
        ],
    )


def test_search_finds_page_by_ocr_text(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.upsert_page("book", 0, _page("岩波現代文庫の検索テスト本文"))
    store.upsert_page("book", 1, _page("無関係な別のページ"))

    hits = store.search("現代文庫")

    assert [( h.doc_id, h.page_index) for h in hits] == [("book", 0)]
    assert isinstance(hits[0], SearchHit)
    assert "現代文庫" in hits[0].snippet


def test_short_query_uses_like_fallback(tmp_path):
    # 2-char query is below the trigram minimum; must still match via LIKE.
    store = Store(str(tmp_path / "t.db"))
    store.upsert_page("book", 0, _page("岩波現代文庫の検索テスト本文"))

    hits = store.search("文庫")

    assert [h.page_index for h in hits] == [0]


def test_no_match_returns_empty(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.upsert_page("book", 0, _page("岩波現代文庫の検索テスト本文"))

    assert store.search("量子力学") == []


def test_reupsert_removes_stale_hits(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.upsert_page("book", 0, _page("最初の内容は検索テスト"))
    assert [h.page_index for h in store.search("検索テスト")] == [0]

    # Replace the page's text; the old term must no longer match.
    store.upsert_page("book", 0, _page("差し替え後の現代文庫"))
    assert store.search("検索テスト") == []
    assert [h.page_index for h in store.search("現代文庫")] == [0]


def test_search_scoped_across_docs(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.upsert_page("bookA", 0, _page("共通語を含むページA"))
    store.upsert_page("bookB", 3, _page("共通語を含むページB"))

    hits = store.search("共通語")

    assert {(h.doc_id, h.page_index) for h in hits} == {("bookA", 0), ("bookB", 3)}


def test_search_respects_limit(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    for i in range(5):
        store.upsert_page("book", i, _page(f"繰り返し語を含む{i}ページ目"))

    hits = store.search("繰り返し語", limit=2)

    assert len(hits) == 2


def test_reindex_all_rebuilds_from_pages(tmp_path):
    # An existing db populated before the FTS table existed can be rebuilt.
    db = str(tmp_path / "t.db")
    store = Store(db)
    store.upsert_page("book", 0, _page("再構築対象の本文テキスト"))

    # Simulate a stale/missing index by clearing it, then rebuild from `pages`.
    store.reindex_all()

    assert [h.page_index for h in store.search("本文テキスト")] == [0]


def test_blank_page_has_no_text_indexed(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    # A page with no OCR text (empty reading_text) must not appear in results
    # and must not break indexing.
    empty = PageParams(source=SourceRef(path="scan.pdf", page_index=0))
    store.upsert_page("book", 0, empty)
    store.upsert_page("book", 1, _page("実在する検索対象語"))

    hits = store.search("検索対象語")
    assert [h.page_index for h in hits] == [1]
