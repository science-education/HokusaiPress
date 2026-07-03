"""WP-72: per-doc ownership (home-server multi-user, complete data separation).

Each doc belongs to exactly one user, assigned once at ingest time. Library/
queue/search listings are filtered to the requesting user's docs by the
webui layer (WP-73) via doc_ids_for_user. Ownership is never silently
reassigned. Written and implemented directly by Claude (delicate, cross-
cutting store.py change), not delegated.
"""

from hokusai_press.store import Store


def test_set_and_get_doc_owner(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.set_doc_owner("book.pdf", "alice")
    assert store.get_doc_owner("book.pdf") == "alice"


def test_unowned_doc_has_no_owner(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    assert store.get_doc_owner("nope.pdf") is None


def test_doc_ids_for_user_is_scoped(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.set_doc_owner("a.pdf", "alice")
    store.set_doc_owner("b.pdf", "alice")
    store.set_doc_owner("c.pdf", "bob")

    assert store.doc_ids_for_user("alice") == ["a.pdf", "b.pdf"]
    assert store.doc_ids_for_user("bob") == ["c.pdf"]
    assert store.doc_ids_for_user("carol") == []


def test_ownership_is_not_silently_reassigned(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.set_doc_owner("book.pdf", "alice")
    # A second claim by a different user must not steal ownership.
    store.set_doc_owner("book.pdf", "bob")
    assert store.get_doc_owner("book.pdf") == "alice"
    assert store.doc_ids_for_user("bob") == []


def test_reclaiming_same_owner_is_a_noop(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.set_doc_owner("book.pdf", "alice")
    store.set_doc_owner("book.pdf", "alice")  # idempotent, no error
    assert store.doc_ids_for_user("alice") == ["book.pdf"]
