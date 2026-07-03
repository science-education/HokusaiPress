"""Contract for WP-73: session auth (login/logout/bootstrap, route protection).

docs/VIEWER_LIBRARY_PLAN.md follow-up (home-server multi-user). Session auth
is opt-in via `create_app(db, require_auth=True)` so every existing test
(which calls create_app(db) with the default False) keeps working unchanged.
When enabled, an HTTP middleware gates every route except the small public
allowlist (login/bootstrap) behind a signed session cookie. Skip until
implemented.
"""

import inspect

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from hokusai_press.webui.app import create_app

if "require_auth" not in inspect.signature(create_app).parameters:
    pytest.skip(
        "WP-73 unimplemented: create_app(db, require_auth=...) missing "
        "(see docs/tasks/WP-73-codex.md)",
        allow_module_level=True,
    )


def test_backward_compat_default_is_open(tmp_path):
    # Every pre-existing test relies on this: no auth required by default.
    client = TestClient(create_app(str(tmp_path / "t.db")))
    assert client.get("/api/queue").status_code == 200


def test_bootstrap_creates_first_admin_and_logs_in(tmp_path):
    client = TestClient(create_app(str(tmp_path / "t.db"), require_auth=True))
    r = client.post("/api/setup", json={"username": "alice", "password": "pw12345"})
    assert r.status_code == 200
    assert r.json()["role"] == "admin"
    # bootstrap logs the new admin in (session cookie set)
    me = client.get("/api/me")
    assert me.status_code == 200
    assert me.json()["username"] == "alice"


def test_bootstrap_fails_once_a_user_exists(tmp_path):
    db = str(tmp_path / "t.db")
    c1 = TestClient(create_app(db, require_auth=True))
    c1.post("/api/setup", json={"username": "alice", "password": "pw12345"})

    c2 = TestClient(create_app(db, require_auth=True))
    r = c2.post("/api/setup", json={"username": "mallory", "password": "x"})
    assert r.status_code == 403


def test_login_wrong_password_is_401(tmp_path):
    db = str(tmp_path / "t.db")
    setup = TestClient(create_app(db, require_auth=True))
    setup.post("/api/setup", json={"username": "alice", "password": "pw12345"})

    client = TestClient(create_app(db, require_auth=True))
    r = client.post("/api/login", json={"username": "alice", "password": "wrong"})
    assert r.status_code == 401


def test_protected_route_401_without_session(tmp_path):
    client = TestClient(create_app(str(tmp_path / "t.db"), require_auth=True))
    r = client.get("/api/queue")
    assert r.status_code == 401


def test_protected_route_allowed_with_session(tmp_path):
    db = str(tmp_path / "t.db")
    client = TestClient(create_app(db, require_auth=True))
    client.post("/api/setup", json={"username": "alice", "password": "pw12345"})

    r = client.get("/api/queue")
    assert r.status_code == 200


def test_logout_clears_session(tmp_path):
    db = str(tmp_path / "t.db")
    client = TestClient(create_app(db, require_auth=True))
    client.post("/api/setup", json={"username": "alice", "password": "pw12345"})
    assert client.get("/api/me").status_code == 200

    client.post("/api/logout")
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/queue").status_code == 401


def test_only_admin_can_create_users(tmp_path):
    db = str(tmp_path / "t.db")
    admin = TestClient(create_app(db, require_auth=True))
    admin.post("/api/setup", json={"username": "alice", "password": "pw12345"})

    r = admin.post("/api/users",
                   json={"username": "bob", "password": "pw67890", "role": "user"})
    assert r.status_code == 200

    bob = TestClient(create_app(db, require_auth=True))
    bob.post("/api/login", json={"username": "bob", "password": "pw67890"})
    r2 = bob.post("/api/users",
                  json={"username": "carol", "password": "x", "role": "user"})
    assert r2.status_code == 403


def test_list_users_excludes_password_hash(tmp_path):
    db = str(tmp_path / "t.db")
    admin = TestClient(create_app(db, require_auth=True))
    admin.post("/api/setup", json={"username": "alice", "password": "pw12345"})

    r = admin.get("/api/users")
    assert r.status_code == 200
    body = r.json()
    assert body[0]["username"] == "alice"
    dumped = str(body)
    assert "pw12345" not in dumped
    assert "password_hash" not in dumped
    assert "salt" not in dumped
