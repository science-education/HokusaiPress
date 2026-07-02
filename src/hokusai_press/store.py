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

CREATE VIRTUAL TABLE IF NOT EXISTS page_search USING fts5(
    doc_id UNINDEXED,
    page_index UNINDEXED,
    text,
    tokenize='trigram'
);

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

-- Parameter-profile tables (docs/PARAM_PROFILE_PLAN.md). Copyright-safe:
-- geometry/structure/statistics only, never OCR text content.
CREATE TABLE IF NOT EXISTS book (
    book_id     TEXT PRIMARY KEY,
    isbn        TEXT,
    title       TEXT,
    author      TEXT,
    publisher   TEXT,
    year        INTEGER,
    ndc         TEXT,
    fmt         TEXT,
    writing_dir TEXT,
    binding     TEXT,
    source      TEXT,
    confidence  REAL,
    resolved_at REAL
);

CREATE TABLE IF NOT EXISTS scan_profile (
    book_id      TEXT PRIMARY KEY,
    scanner_sig  TEXT,
    front_end    INTEGER,
    body_end     INTEGER,
    page_count   INTEGER,
    ocr_pages    INTEGER,
    profile_json TEXT,
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS page_feature (
    book_id      TEXT NOT NULL,
    page_index   INTEGER NOT NULL,
    is_ocr       INTEGER,
    region_count INTEGER,
    deskew_angle REAL,
    deskew_conf  REAL,
    content_w    REAL,
    content_h    REAL,
    nombre_value INTEGER,
    nombre_cx    REAL,
    nombre_cy    REAL,
    blank        INTEGER,
    PRIMARY KEY (book_id, page_index)
);

CREATE TABLE IF NOT EXISTS region_feature (
    book_id    TEXT NOT NULL,
    page_index INTEGER NOT NULL,
    cls        TEXT,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL,
    conf REAL,
    source TEXT
);
CREATE INDEX IF NOT EXISTS idx_region_book ON region_feature(book_id);

-- A bucket key uses '' for an absent scope dimension so the primary key is
-- well-defined (SQLite treats NULLs as distinct in a PK).
CREATE TABLE IF NOT EXISTS bucket_profile (
    scanner    TEXT NOT NULL DEFAULT '',
    fmt        TEXT NOT NULL DEFAULT '',
    genre      TEXT NOT NULL DEFAULT '',
    writing    TEXT NOT NULL DEFAULT '',
    binding    TEXT NOT NULL DEFAULT '',
    param_name TEXT NOT NULL,
    median     REAL,
    mad        REAL,
    n          INTEGER,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scanner, fmt, genre, writing, binding, param_name)
);
"""


@dataclass
class PageRow:
    doc_id: str
    page_index: int
    params: PageParams
    review_status: str
    flags: list[str]


@dataclass
class SearchHit:
    doc_id: str
    page_index: int
    snippet: str


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
            self.conn.execute(
                "DELETE FROM page_search WHERE doc_id=? AND page_index=?",
                (doc_id, page_index),
            )
            text = params.reading_text()
            if text:
                self.conn.execute(
                    "INSERT INTO page_search(doc_id, page_index, text) VALUES(?,?,?)",
                    (doc_id, page_index, text),
                )
            self.conn.commit()

    def search(self, query: str, limit: int = 50) -> list[SearchHit]:
        with self._lock:
            if len(query) >= 3:
                rows = self.conn.execute(
                    "SELECT doc_id, page_index, "
                    "snippet(page_search, 2, '', '', '…', 32) AS snippet "
                    "FROM page_search WHERE page_search MATCH ? "
                    "ORDER BY bm25(page_search) LIMIT ?",
                    (query, limit),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT doc_id, page_index, text FROM page_search "
                    "WHERE text LIKE ? ORDER BY doc_id, page_index LIMIT ?",
                    (f"%{query}%", limit),
                ).fetchall()

        if len(query) >= 3:
            return [SearchHit(r["doc_id"], r["page_index"], r["snippet"]) for r in rows]
        return [
            SearchHit(r["doc_id"], r["page_index"], _like_snippet(r["text"], query))
            for r in rows
        ]

    def reindex_all(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM page_search")
            rows = self.conn.execute(
                "SELECT doc_id, page_index, params_json FROM pages"
            ).fetchall()
            indexed = []
            for row in rows:
                text = _decode_page(json.loads(row["params_json"])).reading_text()
                if text:
                    indexed.append((row["doc_id"], row["page_index"], text))
            self.conn.executemany(
                "INSERT INTO page_search(doc_id, page_index, text) VALUES(?,?,?)",
                indexed,
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

    # --- parameter profile (docs/PARAM_PROFILE_PLAN.md) ---

    def save_book(self, book_id: str, *, isbn=None, title=None, author=None,
                  publisher=None, year=None, ndc=None, fmt=None,
                  writing_dir=None, binding=None,
                  source=None, confidence=None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO book(book_id,isbn,title,author,publisher,year,ndc,"
                "fmt,writing_dir,binding,source,confidence,resolved_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(book_id) DO UPDATE SET isbn=excluded.isbn,"
                "title=excluded.title,author=excluded.author,"
                "publisher=excluded.publisher,year=excluded.year,ndc=excluded.ndc,"
                "fmt=excluded.fmt,writing_dir=excluded.writing_dir,"
                "binding=excluded.binding,source=excluded.source,"
                "confidence=excluded.confidence,resolved_at=excluded.resolved_at",
                (book_id, isbn, title, author, publisher, year, ndc, fmt,
                 writing_dir, binding, source, confidence, time.time()),
            )
            self.conn.commit()

    def list_books(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM book ORDER BY title"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_book(self, book_id: str) -> Optional[dict]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM book WHERE book_id=?", (book_id,)
            ).fetchone()
        return dict(row) if row else None

    def search_books(self, query: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM book WHERE title LIKE ? OR author LIKE ? "
                "OR publisher LIKE ? ORDER BY title",
                (f"%{query}%",) * 3,
            ).fetchall()
        return [dict(r) for r in rows]

    def save_scan_profile(self, book_id: str, scanner_sig, front_end, body_end,
                          page_count, ocr_pages, profile_json: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO scan_profile(book_id,scanner_sig,front_end,body_end,"
                "page_count,ocr_pages,profile_json,created_at) "
                "VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(book_id) DO UPDATE SET scanner_sig=excluded.scanner_sig,"
                "front_end=excluded.front_end,body_end=excluded.body_end,"
                "page_count=excluded.page_count,ocr_pages=excluded.ocr_pages,"
                "profile_json=excluded.profile_json,created_at=excluded.created_at",
                (book_id, scanner_sig, front_end, body_end, page_count,
                 ocr_pages, profile_json, time.time()),
            )
            self.conn.commit()

    def save_page_features(self, book_id: str, feats) -> None:
        rows = [(book_id, f.page_index, int(f.is_ocr), f.region_count,
                 f.deskew_angle, f.deskew_conf, f.content_w, f.content_h,
                 f.nombre_value, f.nombre_cx, f.nombre_cy, int(f.blank))
                for f in feats]
        with self._lock:
            self.conn.execute(
                "DELETE FROM page_feature WHERE book_id=?", (book_id,))
            self.conn.executemany(
                "INSERT INTO page_feature(book_id,page_index,is_ocr,region_count,"
                "deskew_angle,deskew_conf,content_w,content_h,nombre_value,"
                "nombre_cx,nombre_cy,blank) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self.conn.commit()

    def save_region_features(self, book_id: str, feats) -> None:
        rows = [(book_id, f.page_index, f.cls, f.x0, f.y0, f.x1, f.y1,
                 f.conf, f.source) for f in feats]
        with self._lock:
            self.conn.execute(
                "DELETE FROM region_feature WHERE book_id=?", (book_id,))
            self.conn.executemany(
                "INSERT INTO region_feature(book_id,page_index,cls,x0,y0,x1,y1,"
                "conf,source) VALUES(?,?,?,?,?,?,?,?,?)", rows)
            self.conn.commit()

    def page_features(self, book_id: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM page_feature WHERE book_id=? ORDER BY page_index",
                (book_id,)).fetchall()
        return [dict(r) for r in rows]

    def upsert_bucket_param(self, scanner, fmt, genre, writing, binding,
                            param_name: str, median, mad, n) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO bucket_profile(scanner,fmt,genre,writing,binding,"
                "param_name,median,mad,n,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(scanner,fmt,genre,writing,binding,param_name) DO "
                "UPDATE SET median=excluded.median,mad=excluded.mad,n=excluded.n,"
                "updated_at=excluded.updated_at",
                (scanner or "", fmt or "", genre or "", writing or "",
                 binding or "", param_name, median, mad, n, time.time()),
            )
            self.conn.commit()

    def get_bucket_param(self, scanner, fmt, genre, writing, binding,
                         param_name: str):
        with self._lock:
            row = self.conn.execute(
                "SELECT median,mad,n FROM bucket_profile WHERE scanner=? AND "
                "fmt=? AND genre=? AND writing=? AND binding=? AND param_name=?",
                (scanner or "", fmt or "", genre or "", writing or "",
                 binding or "", param_name)).fetchone()
        return (row["median"], row["mad"], row["n"]) if row else None


def _page_to_dict(params: PageParams) -> dict:
    from dataclasses import asdict

    from .model import _enc

    return json.loads(json.dumps(asdict(params), default=_enc))


def _like_snippet(text: str, query: str, context: int = 40) -> str:
    match = text.find(query)
    if match < 0:
        return text[: context * 2]
    start = max(0, match - context)
    end = min(len(text), match + len(query) + context)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def _row_to_page(row: sqlite3.Row) -> PageRow:
    return PageRow(
        doc_id=row["doc_id"],
        page_index=row["page_index"],
        params=_decode_page(json.loads(row["params_json"])),
        review_status=row["review_status"],
        flags=json.loads(row["flags"]),
    )
