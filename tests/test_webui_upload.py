"""Contract for WP-63a: file upload endpoint (drag-and-drop ingest backend).

docs/VIEWER_LIBRARY_PLAN.md follow-up. WP-61a's /api/ingest only accepts a
server-side path. This adds POST /api/upload (multipart) so the browser can
drag-and-drop a file directly: the server saves it under an uploads dir and
runs the same background ingest job as /api/ingest. Skip until implemented.
"""

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient

from hokusai_press.webui.app import create_app

import inspect

if "upload_dir" not in inspect.signature(create_app).parameters:
    pytest.skip(
        "WP-63a unimplemented: create_app(db, upload_dir=...) missing "
        "(see docs/tasks/WP-63a-codex.md)",
        allow_module_level=True,
    )


def _require(client):
    # Probe with a tiny real upload rather than a bad request, so a 404
    # (route missing) is distinguishable from a 4xx validation error.
    r = client.post("/api/upload", files={"file": ("probe.pdf", b"%PDF-x", "application/pdf")})
    if r.status_code == 404:
        pytest.skip("WP-63a unimplemented (see docs/tasks/WP-63a-codex.md)")
    return r


def _wait_status(client, job_id, want, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = {j["job_id"]: j for j in client.get("/api/jobs").json()}
        if job_id in jobs and jobs[job_id]["status"] == want:
            return jobs[job_id]
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {want}")


def test_upload_saves_file_and_starts_job(tmp_path):
    calls = []

    def fake_runner(path, out_pdf, db_path):
        calls.append(path)

    db = str(tmp_path / "t.db")
    client = TestClient(create_app(db, runner=fake_runner, upload_dir=str(tmp_path / "uploads")))

    r = _require(client)
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == "probe.pdf"

    job = _wait_status(client, body["job_id"], "done")
    assert job["error"] is None
    # The runner received a real, existing server-side path.
    assert len(calls) == 1
    import os
    assert os.path.isfile(calls[0])
    with open(calls[0], "rb") as f:
        assert f.read() == b"%PDF-x"


def test_upload_preserves_original_filename_as_doc_id(tmp_path):
    db = str(tmp_path / "t.db")
    client = TestClient(create_app(db, runner=lambda *a: None,
                                    upload_dir=str(tmp_path / "uploads")))
    r = client.post("/api/upload",
                    files={"file": ("スキャン.pdf", b"data", "application/pdf")})
    if r.status_code == 404:
        pytest.skip("WP-63a unimplemented (see docs/tasks/WP-63a-codex.md)")
    assert r.json()["doc_id"] == "スキャン.pdf"


def test_create_app_still_works_without_upload_dir(tmp_path):
    # Backward compatibility: existing callers don't pass upload_dir.
    client = TestClient(create_app(str(tmp_path / "t.db")))
    assert client.get("/api/queue").status_code == 200
