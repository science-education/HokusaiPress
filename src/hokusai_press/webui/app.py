"""Review web UI.

Shows ONLY flagged pages (the ~5% the batch wasn't sure about) — the design
never asks a human to inspect every page. Each correction edits PageParams in
the store and is appended to the decision log with the page's feature vector,
which later trains away the hand-tuned thresholds.

This is a lean v0: queue + per-page view with analysis/output image previews
and a page-kind override that records the decision. Region editing is the next
step; the data flow (queue -> decide -> log -> rebuild) is complete and tested.

Note: this module is imported only in the web context (requires the [web]
extra), so pydantic is imported at module level. It deliberately does NOT use
`from __future__ import annotations` — that would stringify the endpoint type
hints and break FastAPI's body-model resolution.
"""

from typing import Optional

from pydantic import BaseModel

from ..model import DecidedBy, PageKind, ReviewStatus
from ..store import Store


class Decision(BaseModel):
    page_kind: Optional[str] = None
    approve: bool = False
    decided_by: str = "human"


class RegionEdit(BaseModel):
    kind: str               # text | figure | photo
    x0: float
    y0: float
    x1: float
    y1: float
    tone: Optional[str] = None   # photo only: None(auto) | gray | color
    decided_by: str = "human"


def create_app(db_path: str):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response

    app = FastAPI(title="HokusaiPress review")
    store = Store(db_path)

    def _load_original(params):
        from ..source import load_single

        original, _ = load_single(params.source.path, params.source.page_index)
        return original

    @app.get("/", response_class=HTMLResponse)
    def index():
        rows = store.review_queue()
        items = "".join(
            f'<li><a href="/page/{r.doc_id}/{r.page_index}">{r.doc_id} '
            f"p{r.page_index}</a> &mdash; {', '.join(r.flags)}</li>"
            for r in rows
        )
        return (
            "<h2>review queue</h2>"
            f"<p>{len(rows)} pages awaiting review</p>"
            f"<ul>{items or '<li>empty</li>'}</ul>"
        )

    @app.get("/page/{doc_id}/{page_index}", response_class=HTMLResponse)
    def page_view(doc_id: str, page_index: int):
        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        flags = ", ".join(row.flags) or "(none)"
        base = f"/img/{doc_id}/{page_index}"
        return f"""
<h2>{doc_id} &mdash; page {page_index}</h2>
<p>flags: {flags} &middot; status: {row.review_status}</p>
<div style="display:flex;gap:16px;flex-wrap:wrap">
  <div><h3>analysis</h3><img src="{base}/analysis.png" style="max-width:520px;border:1px solid #ccc"></div>
  <div><h3>output</h3><img src="{base}/output.png" style="max-width:520px;border:1px solid #ccc"></div>
</div>
<p>set page kind:
  <button onclick="decide('bw')">bw</button>
  <button onclick="decide('gray')">gray</button>
  <button onclick="decide('color')">color</button>
  <button onclick="approve()">approve as-is</button>
</p>

<details style="max-width:760px;margin:8px 0;padding:8px 12px;background:#f6f6f6;border:1px solid #ddd">
<summary><b>page kind とは？「bw の中のグレー画像」はどうする？</b></summary>
<p style="margin:6px 0">処理は<b>2軸</b>です。混同しないでください。</p>
<ul style="margin:6px 0">
  <li><b>page kind</b>（bw / gray / color）= ページ<b>全体</b>のベース層コーデック。
      <code>gray</code>/<code>color</code> は「全面が写真・図版」「地紙ごと退色」など
      <b>全面トーン維持</b>が要るときの<b>フォールバック</b>です。</li>
  <li><b>region</b>（text / figure / photo）= <b>領域ごと</b>のトーン処理。これが本命。
      <code>text</code>・<code>figure</code>(線画) は二値（くっきり）、
      <code>photo</code>(連続調) はその矩形だけグレー/カラーJPEGで上に重ねます（MRC）。</li>
</ul>
<p style="margin:6px 0"><b>「bw ページの中のグレー写真」→ page kind は bw のまま、
その範囲を <code>photo</code> 領域として下で追加</b>してください。
文字は二値で鮮明・写真だけグレーで軽い、が同一ページ内で両立します。
gray/color は自動判定（領域の彩度）ですが、下の <i>tone</i> で固定もできます。</p>
</details>

<h3>add region（領域ごとの上書き）</h3>
<p>
  kind:
  <select id="rk" onchange="tonevis()">
    <option value="text">text（二値）</option>
    <option value="figure">figure 線画（二値）</option>
    <option value="photo" selected>photo 写真（トーン維持）</option>
  </select>
  <span id="tonewrap">tone:
    <select id="rtone">
      <option value="">auto（彩度で判定）</option>
      <option value="gray">gray 固定</option>
      <option value="color">color 固定</option>
    </select>
  </span>
  <br>
  box (原画像px): x0<input id="x0" size="5"> y0<input id="y0" size="5">
  x1<input id="x1" size="5"> y1<input id="y1" size="5">
  <button onclick="addRegion()">add</button>
</p>
<p style="color:#666;font-size:90%">※ 座標は数値入力（ドラッグ描画は次段）。
解析画像の枠を目安に原画像ピクセルで指定してください。</p>

<p><a href="/">&larr; back to queue</a></p>
<script>
function tonevis(){{
  document.getElementById('tonewrap').style.display =
    document.getElementById('rk').value==='photo' ? 'inline' : 'none';
}}
async function decide(kind){{
  await fetch('/api/page/{doc_id}/{page_index}/decide',{{method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{page_kind:kind}})}});
  location.href='/';
}}
async function approve(){{
  await fetch('/api/page/{doc_id}/{page_index}/decide',{{method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{approve:true}})}});
  location.href='/';
}}
async function addRegion(){{
  const v=id=>parseFloat(document.getElementById(id).value);
  const r=await fetch('/api/page/{doc_id}/{page_index}/region',{{method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{kind:document.getElementById('rk').value,
      tone:document.getElementById('rtone').value||null,
      x0:v('x0'),y0:v('y0'),x1:v('x1'),y1:v('y1')}})}});
  const b=await r.json(); alert('added. regions='+b.regions); location.href='/';
}}
tonevis();
</script>"""

    @app.get("/img/{doc_id}/{page_index}/analysis.png")
    def analysis_png(doc_id: str, page_index: int):
        from ..preview import analysis_overlay

        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        png = analysis_overlay(_load_original(row.params), row.params)
        return Response(content=png, media_type="image/png")

    @app.get("/img/{doc_id}/{page_index}/output.png")
    def output_png(doc_id: str, page_index: int):
        from ..model import RenderSettings
        from ..preview import output_preview

        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        png = output_preview(_load_original(row.params), row.params, RenderSettings())
        return Response(content=png, media_type="image/png")

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
        from ..learn import page_features

        params = row.params
        features = page_features(params)
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

    @app.post("/api/page/{doc_id}/{page_index}/region")
    def add_region(doc_id: str, page_index: int, edit: RegionEdit):
        from ..learn import page_features
        from ..model import Box, Region, RegionKind

        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        params = row.params
        params.regions.append(Region(
            kind=RegionKind(edit.kind),
            box=Box(edit.x0, edit.y0, edit.x1, edit.y1),
            source="manual",
            tone=edit.tone if edit.kind == "photo" else None,
        ))
        params.review_status = ReviewStatus.CORRECTED
        params.decided_by = DecidedBy(edit.decided_by)
        store.log_decision(doc_id, page_index, edit.decided_by, "region",
                           None, {"kind": edit.kind, "tone": edit.tone,
                                  "box": [edit.x0, edit.y0, edit.x1, edit.y1]},
                           page_features(params))
        store.upsert_page(doc_id, page_index, params)
        return {"status": params.review_status.value,
                "regions": len(params.regions)}

    return app


def serve(db_path: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    app = create_app(db_path)
    print(f"HokusaiPress review UI -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
