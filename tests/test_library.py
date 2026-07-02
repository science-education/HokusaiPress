"""Contract for WP-02: book metadata reads + `search` CLI (docs/VIEWER_LIBRARY_PLAN.md).

Written BEFORE implementation. Book rows are saved today via Store.save_book;
this WP adds the read side (list/get/search over book metadata) plus a CLI
`search` subcommand that surfaces the WP-01 full-text index. Skip until the
read API lands so the suite stays green.
"""

import pytest

from hokusai_press.model import (
    Box,
    PageParams,
    Region,
    RegionKind,
    SourceRef,
)
from hokusai_press.store import Store

# Pending contract: activate once WP-02 adds the book-read API.
_probe = Store(":memory:")
if not hasattr(_probe, "search_books"):
    _probe.close()
    pytest.skip(
        "WP-02 unimplemented: Store.list_books/get_book/search_books missing "
        "(see docs/tasks/WP-02-codex.md)",
        allow_module_level=True,
    )
_probe.close()


def _seed_books(db):
    store = Store(db)
    store.save_book("bookA", title="吾輩は猫である", author="夏目漱石",
                    publisher="岩波書店", year=1905)
    store.save_book("bookB", title="銀河鉄道の夜", author="宮沢賢治",
                    publisher="新潮社", year=1934)
    return store


def _page(text: str, idx: int = 0) -> PageParams:
    return PageParams(
        source=SourceRef(path="scan.pdf", page_index=idx),
        regions=[Region(kind=RegionKind.TEXT, box=Box(0, 0, 100, 100),
                        ocr_text=text)],
    )


def test_list_books_ordered_by_title(tmp_path):
    store = _seed_books(str(tmp_path / "t.db"))
    books = store.list_books()
    assert [b["title"] for b in books] == ["吾輩は猫である", "銀河鉄道の夜"]
    assert books[0]["author"] == "夏目漱石"


def test_get_book(tmp_path):
    store = _seed_books(str(tmp_path / "t.db"))
    assert store.get_book("bookB")["author"] == "宮沢賢治"
    assert store.get_book("nope") is None


def test_search_books_matches_title_and_author(tmp_path):
    store = _seed_books(str(tmp_path / "t.db"))
    assert [b["book_id"] for b in store.search_books("漱石")] == ["bookA"]
    assert [b["book_id"] for b in store.search_books("銀河")] == ["bookB"]
    assert store.search_books("存在しない") == []


def test_cli_search_prints_page_hits(tmp_path, capsys):
    from hokusai_press.cli import main

    db = str(tmp_path / "t.db")
    store = Store(db)
    store.upsert_page("book.pdf", 4, _page("量子力学の基礎を解説する本文", idx=4))
    store.close()

    rc = main(["search", "量子力学", "--db", db])
    assert rc == 0
    out = capsys.readouterr().out
    # The hit line names the doc and page so the reader can jump there.
    assert "book.pdf" in out
    assert "4" in out
