"""Review web UI.

Shows ONLY flagged pages (the ~5% the batch wasn't sure about) — the design
never asks a human to inspect every page. Each correction edits PageParams in
the store and is appended to the decision log with the page's feature vector,
which later trains away the hand-tuned thresholds.

This is a lean v0: queue list + per-page JSON + a page-kind override endpoint
that records the decision. Image previews and region editing are wired as the
next step; the data flow (queue -> decide -> log) is complete and testable.
"""

from __future__ import annotations

from typing import Optional

from ..model import DecidedBy, PageKind, ReviewStatus
from ..store import Store


def create_app(db_path: str):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    app = FastAPI(title="HokusaiPress review")
    store = Store(db_path)

    class Decision(BaseModel):
        page_kind: Optional[str] = None
        approve: bool = False
        decided_by: str = "human"

    @app.get("/", response_class=HTMLResponse)
    def index():
        rows = store.review_queue()
        items = "".join(
            f'<li><a href="/page/{r.doc_id}/{r.page_index}">{r.doc_id} '
            f"p{r.page_index}</a> — {', '.join(r.flags)}</li>"
            for r in rows
        )
        return (
            "<h2>review queue</h2>"
            f"<p>{len(rows)} pages awaiting review</p>"
            f"<ul>{items or '<li>empty</li>'}</ul>"
        )

    @app.get("/api/queue")
    def queue():
        return [
            {"doc_id": r.doc_id, "page_index": r.page_index, "flags": r.flags}
            for r in store.review_queue()
        ]

    @app.get("/api/page/{doc_id}/{page_index}")
    def get_page(doc_id: str, page_index: int):
        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        from dataclasses import asdict

        from ..model import _enc
        import json

        return json.loads(json.dumps(asdict(row.params), default=_enc))

    @app.post("/api/page/{doc_id}/{page_index}/decide")
    def decide(doc_id: str, page_index: int, decision: Decision):
        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        params = row.params
        features = _page_features(params)
        if decision.page_kind:
            old = params.page_kind.value
            params.page_kind = PageKind(decision.page_kind)
            params.review_status = ReviewStatus.CORRECTED
            params.decided_by = DecidedBy(decision.decided_by)
            store.log_decision(doc_id, page_index, decision.decided_by,
                               "page_kind", old, decision.page_kind, features)
        elif decision.approve:
            params.review_status = ReviewStatus.APPROVED
            params.decided_by = DecidedBy(decision.decided_by)
            store.log_decision(doc_id, page_index, decision.decided_by,
                               "approve", None, True, features)
        store.upsert_page(doc_id, page_index, params)
        return {"status": params.review_status.value}

    return app


def _page_features(params) -> dict:
    """Feature vector logged with each decision (training data for later)."""
    return {
        "deskew_angle": params.deskew.angle_deg,
        "deskew_conf": params.deskew.confidence,
        "margin_conf": params.margin.confidence if params.margin else 0.0,
        "n_text": sum(1 for r in params.regions if r.kind.value == "text"),
        "n_photo": sum(1 for r in params.regions if r.kind.value == "photo"),
        "flags": [f.value for f in params.flags],
    }


def serve(db_path: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    app = create_app(db_path)
    print(f"HokusaiPress review UI -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
