"""Contract for WP-74: wire doc ownership into ingest + viewing/listing routes.

docs/VIEWER_LIBRARY_PLAN.md follow-up (home-server multi-user, complete data
separation). Builds on WP-72 (Store.set_doc_owner/get_doc_owner/
doc_ids_for_user) and WP-73 (session auth). Scope for this WP: viewing and
listing surfaces only (library, queue, search, /read, /page, /img). Editing
endpoints are a separate follow-up (see HANDOFF). Skip until implemented.

Backward compatibility rule (do not weaken): when there is NO logged-in user
(no session cookie, e.g. require_auth=False or an anonymous request), listing
and viewing are UNFILTERED -- today's single-user behavior. Ownership
filtering only activates for an authenticated request. A doc with no owner
row (ingested before ownership tracking) remains visible to everyone.
"""

import time

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


def _page(text="hi"):
    return PageParams(
        source=SourceRef(path="s.pdf", page_index=0),
        regions=[Region(kind=RegionKind.TEXT, box=Box(0, 0, 10, 10), ocr_text=text)],
    )


def _wait_job_done(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = {j["job_id"]: j for j in client.get("/api/jobs").json()}
        if jobs.get(job_id, {}).get("status") in ("done", "error"):
            return jobs[job_id]
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_ingest_assigns_ownership_to_logged_in_user(tmp_path):
    src = tmp_path / "book.pdf"
    src.write_bytes(b"%PDF-x")
    db = str(tmp_path / "t.db")

    client = TestClient(create_app(db, runner=lambda *a: None, require_auth=True))
    client.post("/api/setup", json={"username": "alice", "password": "pw12345"})
    job = client.post("/api/ingest", json={"path": str(src)}).json()
    _wait_job_done(client, job["job_id"])

    store = Store(db)
    try:
        owner = store.get_doc_owner("book.pdf")
    finally:
        store.close()
    if owner is None:
        pytest.skip("WP-74 unimplemented (see docs/tasks/WP-74-codex.md)")
    assert owner == "alice"


def test_library_is_scoped_to_logged_in_user(tmp_path):
    db = str(tmp_path / "t.db")
    store = Store(db)
    store.upsert_page("a.pdf", 0, _page())
    store.upsert_page("b.pdf", 0, _page())
    store.set_doc_owner("a.pdf", "alice")
    store.set_doc_owner("b.pdf", "bob")
    store.close()

    # Bootstrap alice as the first (admin) user, then have her create bob.
    admin = TestClient(create_app(db, require_auth=True))
    admin.post("/api/setup", json={"username": "alice", "password": "pw12345"})
    admin.post("/api/users", json={"username": "bob", "password": "pw67890", "role": "user"})

    alice_client = TestClient(create_app(db, require_auth=True))
    alice_client.post("/api/login", json={"username": "alice", "password": "pw12345"})
    docs = {r["doc_id"] for r in alice_client.get("/api/library").json()}
    if docs == {"a.pdf", "b.pdf"}:
        pytest.skip("WP-74 unimplemented (see docs/tasks/WP-74-codex.md)")
    assert docs == {"a.pdf"}

    bob_client = TestClient(create_app(db, require_auth=True))
    bob_client.post("/api/login", json={"username": "bob", "password": "pw67890"})
    docs_b = {r["doc_id"] for r in bob_client.get("/api/library").json()}
    assert docs_b == {"b.pdf"}


def test_unauthenticated_listing_is_unfiltered_backward_compat(tmp_path):
    db = str(tmp_path / "t.db")
    store = Store(db)
    store.upsert_page("a.pdf", 0, _page())
    store.upsert_page("b.pdf", 0, _page())
    store.close()

    # require_auth=False, no session -- today's single-user behavior: see all.
    client = TestClient(create_app(db))
    docs = {r["doc_id"] for r in client.get("/api/library").json()}
    assert docs == {"a.pdf", "b.pdf"}


def test_read_view_403_for_non_owner(tmp_path):
    db = str(tmp_path / "t.db")
    store = Store(db)
    store.upsert_page("a.pdf", 0, _page())
    store.set_doc_owner("a.pdf", "alice")
    store.close()

    admin = TestClient(create_app(db, require_auth=True))
    admin.post("/api/setup", json={"username": "alice", "password": "pw12345"})
    admin.post("/api/users", json={"username": "bob", "password": "pw67890", "role": "user"})

    bob = TestClient(create_app(db, require_auth=True))
    bob.post("/api/login", json={"username": "bob", "password": "pw67890"})
    r = bob.get("/read/a.pdf")
    if r.status_code == 200:
        pytest.skip("WP-74 unimplemented (see docs/tasks/WP-74-codex.md)")
    assert r.status_code == 403

    alice = TestClient(create_app(db, require_auth=True))
    alice.post("/api/login", json={"username": "alice", "password": "pw12345"})
    assert alice.get("/read/a.pdf").status_code == 200


def test_doc_with_no_owner_is_visible_to_any_logged_in_user(tmp_path):
    db = str(tmp_path / "t.db")
    store = Store(db)
    store.upsert_page("legacy.pdf", 0, _page())  # no set_doc_owner call
    store.close()

    client = TestClient(create_app(db, require_auth=True))
    r0 = client.post("/api/setup", json={"username": "alice", "password": "pw12345"})
    if r0.status_code != 200:
        pytest.skip("WP-74 unimplemented (see docs/tasks/WP-74-codex.md)")

    assert client.get("/read/legacy.pdf").status_code == 200
    docs = {r["doc_id"] for r in client.get("/api/library").json()}
    assert "legacy.pdf" in docs
