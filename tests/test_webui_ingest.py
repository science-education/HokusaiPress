"""Contract for WP-61a: ingest API — register a scan/PDF from the web UI.

docs/VIEWER_LIBRARY_PLAN.md Phase 1 extension (登録→修整→閲覧 loop). The UI
posts a server-side file path; the app runs the batch pipeline in a background
thread and exposes job status for polling. The heavy pipeline is injectable
(`create_app(db, runner=...)`) so this contract runs without OCR models.
Skip until implemented.
"""

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from hokusai_press.webui.app import create_app


def _require(client):
    if client.get("/api/jobs").status_code != 200:
        pytest.skip("WP-61a unimplemented (see docs/tasks/WP-61a-codex.md)")


def _wait_status(client, job_id, want, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = {j["job_id"]: j for j in client.get("/api/jobs").json()}
        if job_id in jobs and jobs[job_id]["status"] == want:
            return jobs[job_id]
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {want}: {jobs}")


def test_ingest_runs_runner_and_reports_done(tmp_path):
    src = tmp_path / "book.pdf"
    src.write_bytes(b"%PDF-fake")
    calls = []

    def fake_runner(path, out_pdf, db_path):
        calls.append((path, out_pdf, db_path))

    db = str(tmp_path / "t.db")
    client = TestClient(create_app(db, runner=fake_runner))
    _require(client)

    r = client.post("/api/ingest", json={"path": str(src)})
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] == "book.pdf"

    job = _wait_status(client, body["job_id"], "done")
    assert job["doc_id"] == "book.pdf"
    assert job["error"] is None
    # The runner got the source path and the shared db path.
    assert len(calls) == 1
    assert calls[0][0] == str(src)
    assert calls[0][2] == db


def test_ingest_missing_file_is_400(tmp_path):
    client = TestClient(create_app(str(tmp_path / "t.db"), runner=lambda *a: None))
    _require(client)
    r = client.post("/api/ingest", json={"path": str(tmp_path / "nope.pdf")})
    assert r.status_code == 400


def test_ingest_failure_surfaces_error(tmp_path):
    src = tmp_path / "bad.pdf"
    src.write_bytes(b"x")

    def boom(path, out_pdf, db_path):
        raise RuntimeError("ocr exploded")

    client = TestClient(create_app(str(tmp_path / "t.db"), runner=boom))
    _require(client)
    job_id = client.post("/api/ingest", json={"path": str(src)}).json()["job_id"]

    job = _wait_status(client, job_id, "error")
    assert "ocr exploded" in job["error"]


def test_create_app_still_works_without_runner(tmp_path):
    # Backward compatibility: every existing test calls create_app(db) only.
    client = TestClient(create_app(str(tmp_path / "t.db")))
    assert client.get("/api/queue").status_code == 200
