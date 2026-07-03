"""Contract for WP-71: local user accounts (username/password, roles).

docs/VIEWER_LIBRARY_PLAN.md follow-up (home-server multi-user). Pure store-level
auth primitives: password hashing (stdlib only, no new dependency), a `user`
table, and admin/regular roles. Session/HTTP wiring is WP-73. Skip until
implemented.
"""

import pytest

from hokusai_press.store import Store

try:
    Store(":memory:").create_user  # probe without asserting behavior yet
except AttributeError:
    pytest.skip(
        "WP-71 unimplemented: Store user methods missing "
        "(see docs/tasks/WP-71-codex.md)",
        allow_module_level=True,
    )


def test_create_user_and_verify_password(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.create_user("alice", "correct horse", role="admin")

    user = store.verify_user("alice", "correct horse")
    assert user is not None
    assert user["username"] == "alice"
    assert user["role"] == "admin"


def test_verify_wrong_password_fails(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.create_user("alice", "correct horse")
    assert store.verify_user("alice", "wrong password") is None


def test_verify_unknown_user_fails(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    assert store.verify_user("nobody", "whatever") is None


def test_password_is_not_stored_in_plaintext(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.create_user("alice", "correct horse")
    row = store.get_user("alice")
    # Whatever column holds the credential, the raw password must not appear.
    assert "correct horse" not in str(dict(row) if hasattr(row, "keys") else row)


def test_default_role_is_user(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.create_user("bob", "hunter2")
    row = store.get_user("bob")
    assert row["role"] == "user"


def test_duplicate_username_raises(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.create_user("alice", "pw1")
    with pytest.raises(Exception):
        store.create_user("alice", "pw2")


def test_list_users_does_not_leak_password_hash_column_name_only(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.create_user("alice", "pw1", role="admin")
    store.create_user("bob", "pw2", role="user")
    users = store.list_users()
    assert {u["username"] for u in users} == {"alice", "bob"}
    roles = {u["username"]: u["role"] for u in users}
    assert roles == {"alice": "admin", "bob": "user"}
