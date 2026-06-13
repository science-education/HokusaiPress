"""SQLite-backed job and decision store.

Two tables:
- pages: one row per page, holding the serialized PageParams and review state,
  so the batch run, the review UI, and the renderer all share one source of
  truth that survives restarts.
- decisions: an append-only log of every human/AI override, with the page
  feature vector captured at decision time. This log is the training data
  that later replaces the hand-tuned auto thresholds with a learned model.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .model import PageParams, _decode_page

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    doc_id        TEXT NOT NULL,
    page_index    INTEGER NOT NULL,
    params_json   TEXT NOT NULL,
    review_status TEXT NOT NULL,
    flags         TEXT NOT NULL,
    updated_at    REAL NOT NULL,
    PRIMARY KEY (doc_id, page_index)
);
CREATE INDEX IF NOT EXISTS idx_pages_status ON pages(review_status);

CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL,
    page_index  INTEGER NOT NULL,
    decided_by  TEXT NOT NULL,
    field       TEXT NOT NULL,
    old_value   TEXT,
    new_value   TEXT,
    features    TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


@dataclass
class PageRow:
    doc_id: str
    page_index: int
    params: PageParams
    review_status: str
    flags: list[str]


class Store:
    def __init__(self, db_path: str = "hokusai.db"):
        # check_same_thread=False lets the review web UI serve sync endpoints
        # from a threadpool. A single page load fires several requests at once
        # (analysis.png + output.png + api calls), so concurrent access to the
        # one connection is real: a lock serializes every statement+fetch so
        # interleaved cursors can't return half-populated rows.
        # sqlite creates the db file on connect but not missing parent dirs, so
        # make them first (skip in-memory / bare-filename paths).
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def upsert_page(self, doc_id: str, page_index: int, params: PageParams) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO pages(doc_id, page_index, params_json, review_status, "
                "flags, updated_at) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(doc_id, page_index) DO UPDATE SET "
                "params_json=excluded.params_json, review_status=excluded.review_status, "
                "flags=excluded.flags, updated_at=excluded.updated_at",
                (
                    doc_id,
                    page_index,
                    json.dumps(_page_to_dict(params), ensure_ascii=False),
                    params.review_status.value,
                    json.dumps([f.value for f in params.flags]),
                    time.time(),
                ),
            )
            self.conn.commit()

    def get_page(self, doc_id: str, page_index: int) -> Optional[PageRow]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM pages WHERE doc_id=? AND page_index=?",
                (doc_id, page_index),
            ).fetchone()
        return _row_to_page(row) if row else None

    def review_queue(self, doc_id: Optional[str] = None) -> list[PageRow]:
        with self._lock:
            if doc_id:
                rows = self.conn.execute(
                    "SELECT * FROM pages WHERE review_status='needs_review' AND doc_id=? "
                    "ORDER BY page_index",
                    (doc_id,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM pages WHERE review_status='needs_review' "
                    "ORDER BY doc_id, page_index"
                ).fetchall()
        return [_row_to_page(r) for r in rows]

    def list_pages(self, doc_id: str) -> list[PageRow]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM pages WHERE doc_id=? ORDER BY page_index", (doc_id,)
            ).fetchall()
        return [_row_to_page(r) for r in rows]

    def doc_ids(self) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT DISTINCT doc_id FROM pages ORDER BY doc_id"
            ).fetchall()
        return [r["doc_id"] for r in rows]

    def log_decision(
        self,
        doc_id: str,
        page_index: int,
        decided_by: str,
        field: str,
        old_value,
        new_value,
        features: dict,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO decisions(doc_id, page_index, decided_by, field, "
                "old_value, new_value, features, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    page_index,
                    decided_by,
                    field,
                    json.dumps(old_value, ensure_ascii=False, default=str),
                    json.dumps(new_value, ensure_ascii=False, default=str),
                    json.dumps(features, ensure_ascii=False, default=str),
                    time.time(),
                ),
            )
            self.conn.commit()

    def decisions_for_training(self, field: Optional[str] = None) -> list[dict]:
        """Return (features, new_value) pairs for the learning step."""
        with self._lock:
            if field:
                rows = self.conn.execute(
                    "SELECT features, new_value FROM decisions WHERE field=?", (field,)
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT field, features, new_value FROM decisions"
                ).fetchall()
        return [dict(r) for r in rows]


def _page_to_dict(params: PageParams) -> dict:
    from dataclasses import asdict

    from .model import _enc

    return json.loads(json.dumps(asdict(params), default=_enc))


def _row_to_page(row: sqlite3.Row) -> PageRow:
    return PageRow(
        doc_id=row["doc_id"],
        page_index=row["page_index"],
        params=_decode_page(json.loads(row["params_json"])),
        review_status=row["review_status"],
        flags=json.loads(row["flags"]),
    )
