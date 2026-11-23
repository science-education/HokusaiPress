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

    def replace_doc_pages(self, doc_id: str, pages: list[PageParams]) -> None:
        """Upsert all pages for *doc_id* and prune any stale rows with
        page_index >= len(pages) in a single transaction."""
        now = time.time()
        n = len(pages)
        rows = [
            (
                doc_id,
                p.source.page_index,
                json.dumps(_page_to_dict(p), ensure_ascii=False),
                p.review_status.value,
                json.dumps([f.value for f in p.flags]),
                now,
            )
            for p in pages
        ]
        with self._lock:
            with self.conn:
                self.conn.executemany(
                    "INSERT INTO pages(doc_id, page_index, params_json, review_status, "
                    "flags, updated_at) VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(doc_id, page_index) DO UPDATE SET "
                    "params_json=excluded.params_json, review_status=excluded.review_status, "
                    "flags=excluded.flags, updated_at=excluded.updated_at",
                    rows,
                )
                self.conn.execute(
                    "DELETE FROM pages WHERE doc_id=? AND page_index>=?",
                    (doc_id, n),
                )
                self.conn.execute(
                    "DELETE FROM page_feature WHERE book_id=? AND page_index>=?",
                    (doc_id, n),
                )
                self.conn.execute(
                    "DELETE FROM region_feature WHERE book_id=? AND page_index>=?",
                    (doc_id, n),
                )

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

    def merge_from(self, other: "Store", full_run: bool = True) -> None:
        """Fold *other* store's tables into this store inside a single transaction.

        When *full_run* is True (whole-book analysis), replaces each doc's pages
        via replace_doc_pages (pruning stale trailing rows) and copies the
        per-book profile tables (book, scan_profile, page_feature, region_feature).
        When *full_run* is False (--pages subset), only upserts the processed
        pages and leaves profile tables untouched.
        Any rows in decisions and bucket_profile present in *other* are also
        merged.
        """
        doc_ids = other.doc_ids()
        pages_by_doc = {doc: other.list_pages(doc) for doc in doc_ids}
        with other._lock:
            books = other.conn.execute("SELECT * FROM book").fetchall()
            scan_profiles = other.conn.execute("SELECT * FROM scan_profile").fetchall()
            page_feats = other.conn.execute("SELECT * FROM page_feature").fetchall()
            region_feats = other.conn.execute("SELECT * FROM region_feature").fetchall()
            decisions = other.conn.execute(
                "SELECT doc_id, page_index, decided_by, field, old_value, "
                "new_value, features, created_at FROM decisions ORDER BY id"
            ).fetchall()
            buckets = other.conn.execute("SELECT * FROM bucket_profile").fetchall()

        with self._lock:
            for doc, rows in pages_by_doc.items():
                if full_run:
                    self.replace_doc_pages(doc, [r.params for r in rows])
                else:
                    for row in rows:
                        self.upsert_page(doc, row.page_index, row.params)

            with self.conn:
                if full_run:
                    for b in books:
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
                            (
                                b["book_id"], b["isbn"], b["title"], b["author"],
                                b["publisher"], b["year"], b["ndc"], b["fmt"],
                                b["writing_dir"], b["binding"], b["source"],
                                b["confidence"], b["resolved_at"],
                            ),
                        )
                    for sp in scan_profiles:
                        self.conn.execute(
                            "INSERT INTO scan_profile(book_id,scanner_sig,front_end,body_end,"
                            "page_count,ocr_pages,profile_json,created_at) "
                            "VALUES(?,?,?,?,?,?,?,?) "
                            "ON CONFLICT(book_id) DO UPDATE SET scanner_sig=excluded.scanner_sig,"
                            "front_end=excluded.front_end,body_end=excluded.body_end,"
                            "page_count=excluded.page_count,ocr_pages=excluded.ocr_pages,"
                            "profile_json=excluded.profile_json,created_at=excluded.created_at",
                            (
                                sp["book_id"], sp["scanner_sig"], sp["front_end"],
                                sp["body_end"], sp["page_count"], sp["ocr_pages"],
                                sp["profile_json"], sp["created_at"],
                            ),
                        )
                    pf_book_ids = {b["book_id"] for b in books} | {pf["book_id"] for pf in page_feats} | set(doc_ids)
                    for bid in pf_book_ids:
                        self.conn.execute("DELETE FROM page_feature WHERE book_id=?", (bid,))
                    if page_feats:
                        self.conn.executemany(
                            "INSERT INTO page_feature(book_id,page_index,is_ocr,region_count,"
                            "deskew_angle,deskew_conf,content_w,content_h,nombre_value,"
                            "nombre_cx,nombre_cy,blank) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            [
                                (
                                    pf["book_id"], pf["page_index"], pf["is_ocr"],
                                    pf["region_count"], pf["deskew_angle"], pf["deskew_conf"],
                                    pf["content_w"], pf["content_h"], pf["nombre_value"],
                                    pf["nombre_cx"], pf["nombre_cy"], pf["blank"],
                                )
                                for pf in page_feats
                            ],
                        )
                    rf_book_ids = {b["book_id"] for b in books} | {rf["book_id"] for rf in region_feats} | set(doc_ids)
                    for bid in rf_book_ids:
                        self.conn.execute("DELETE FROM region_feature WHERE book_id=?", (bid,))
                    if region_feats:
                        self.conn.executemany(
                            "INSERT INTO region_feature(book_id,page_index,cls,x0,y0,x1,y1,"
                            "conf,source) VALUES(?,?,?,?,?,?,?,?,?)",
                            [
                                (
                                    rf["book_id"], rf["page_index"], rf["cls"],
                                    rf["x0"], rf["y0"], rf["x1"], rf["y1"],
                                    rf["conf"], rf["source"],
                                )
                                for rf in region_feats
                            ],
                        )

                if decisions:
                    self.conn.executemany(
                        "INSERT INTO decisions(doc_id, page_index, decided_by, field, "
                        "old_value, new_value, features, created_at) VALUES(?,?,?,?,?,?,?,?)",
                        [
                            (
                                d["doc_id"], d["page_index"], d["decided_by"], d["field"],
                                d["old_value"], d["new_value"], d["features"], d["created_at"],
                            )
                            for d in decisions
                        ],
                    )

                for bp in buckets:
                    self.conn.execute(
                        "INSERT INTO bucket_profile(scanner,fmt,genre,writing,binding,"
                        "param_name,median,mad,n,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(scanner,fmt,genre,writing,binding,param_name) DO "
                        "UPDATE SET median=excluded.median,mad=excluded.mad,n=excluded.n,"
                        "updated_at=excluded.updated_at",
                        (
                            bp["scanner"], bp["fmt"], bp["genre"], bp["writing"],
                            bp["binding"], bp["param_name"], bp["median"],
                            bp["mad"], bp["n"], bp["updated_at"],
                        ),
                    )


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
