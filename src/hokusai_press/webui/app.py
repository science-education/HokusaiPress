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
    finish: bool = False          # finalize edits: mark corrected, leave queue
    decided_by: str = "human"


class RegionEdit(BaseModel):
    kind: str               # text | figure | photo
    x0: float
    y0: float
    x1: float
    y1: float
    tone: Optional[str] = None   # photo only: None(auto) | gray | color
    decided_by: str = "human"


class BoxEdit(BaseModel):       # for content / nombre overrides
    x0: float
    y0: float
    x1: float
    y1: float
    decided_by: str = "human"


class ReaderReport(BaseModel):
    decided_by: str


def create_app(db_path: str):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response

    app = FastAPI(title="HokusaiPress review")
    store = Store(db_path)

    # Small cache of extracted originals: a single page view fires analysis.png
    # and output.png back to back, both needing the same original. Re-extracting
    # (a pdfium render) twice per view is wasteful and adds lock contention, so
    # keep the few most-recent originals keyed by (path, page_index).
    from collections import OrderedDict

    _orig_cache: "OrderedDict[tuple, object]" = OrderedDict()
    _ORIG_CACHE_MAX = 4

    def _load_original(params):
        from ..source import load_single

        key = (params.source.path, params.source.page_index)
        if key in _orig_cache:
            _orig_cache.move_to_end(key)
            return _orig_cache[key]
        original, _ = load_single(*key)
        _orig_cache[key] = original
        while len(_orig_cache) > _ORIG_CACHE_MAX:
            _orig_cache.popitem(last=False)
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
        base = f"/img/{doc_id}/{page_index}"
        # deskewed-original dimensions (= original; deskew keeps the canvas
        # size), so the browser can map a drag back to region pixels.
        oh, ow = _load_original(row.params).shape[:2]
        regions_html = "".join(
            f"<li>#{i} <b>{r.kind.value}</b>"
            + (f"/{r.tone}" if r.tone else "")
            + f" [{int(r.box.x0)},{int(r.box.y0)},{int(r.box.x1)},{int(r.box.y1)}]"
            + f" <i>({r.source})</i>"
            + f' <button onclick="delRegion({i})">削除</button></li>'
            for i, r in enumerate(row.params.regions)
        )
        # counts by class (text / figure / photo / table ...) for an at-a-glance
        # summary above the list
        import collections as _c
        _kc = _c.Counter(r.kind.value for r in row.params.regions)
        kinds_html = " ".join(f"{k}:{v}" for k, v in sorted(_kc.items())) or "(none)"
        # clickable hit areas over each region (positioned in % so they track
        # the image at any display scale); double/right-click deletes. The solid
        # colored boxes themselves come from the analysis PNG underneath.
        from ..preview import (
            _COLORS, _CONTENT_COLOR, _CROP_COLOR, _NOMBRE_COLOR, _bgr_to_hex,
        )

        kind_hex = {k.value: _bgr_to_hex(c) for k, c in _COLORS.items()}

        def _box_div(b, color, handler, label, group):
            # data-group lets setMode() enable hover/click on only the boxes
            # matching the active button, so a big content box on top never
            # blocks selecting a photo/nombre box underneath.
            return (
                f'<div class="rbox" data-group="{group}" title="{label}"'
                f' style="left:{b.x0 / ow * 100:.3f}%;top:{b.y0 / oh * 100:.3f}%;'
                f'width:{max(0.0, b.x1 - b.x0) / ow * 100:.3f}%;'
                f'height:{max(0.0, b.y1 - b.y0) / oh * 100:.3f}%;'
                f'--c:{color}" ondblclick="{handler}"></div>'
            )

        regions_overlay = "".join(
            _box_div(
                r.box, kind_hex.get(r.kind.value, "#888"), f"delRegion({i})",
                r.kind.value + (f"/{r.tone}" if r.tone else "") + " — ダブルクリックで削除",
                r.kind.value,
            )
            for i, r in enumerate(row.params.regions)
        )
        mg = row.params.margin
        if mg and mg.content:
            regions_overlay += _box_div(
                mg.content, _bgr_to_hex(_CONTENT_COLOR), "delBox('content')",
                "内容領域 — ダブルクリックで削除(自動検出に戻す)", "content")
        if mg and mg.nombre_box:
            regions_overlay += _box_div(
                mg.nombre_box, _bgr_to_hex(_NOMBRE_COLOR), "delBox('nombre')",
                "ページ番号領域 — ダブルクリックで削除", "nombre")

        # which box kinds exist on this page (drives the colored legend buttons)
        has = {
            "photo": any(r.kind.value == "photo" for r in row.params.regions),
            "content": bool(mg and mg.content),
            "crop": bool(mg and mg.crop),
            "nombre": bool(mg and mg.nombre_box),
        }

        def _mode_btn(bid, mode, label, hexc, group):
            present = has[group]
            dim = "" if present else "opacity:.45;"
            na = "" if present else "（なし）"
            tip = "" if present else " title='この枠はありません（ドラッグで追加可）'"
            return (f'<button id="{bid}" onclick="setMode(\'{mode}\')"{tip} '
                    f'style="{dim}border-left:7px solid {hexc}">{label}{na}</button>')

        mode_buttons = (
            _mode_btn("mColor", "color", "写真（カラー）", kind_hex["photo"], "photo")
            + _mode_btn("mGray", "gray", "写真（グレー）", kind_hex["photo"], "photo")
            + "&nbsp;"
            + _mode_btn("mContent", "content", "内容枠",
                        _bgr_to_hex(_CONTENT_COLOR), "content")
            + _mode_btn("mNombre", "nombre", "ノンブル",
                        _bgr_to_hex(_NOMBRE_COLOR), "nombre")
        )
        # crop is derived (not editable): show it in the legend only
        _crop_dim = "" if has["crop"] else "opacity:.45;"
        crop_chip = (
            f'<span style="{_crop_dim}display:inline-flex;align-items:center;'
            f'margin-left:8px"><span style="width:13px;height:13px;'
            f'background:{_bgr_to_hex(_CROP_COLOR)};border:1px solid #888;'
            f'margin-right:4px"></span>出力枠{"" if has["crop"] else "（なし）"}</span>'
        )
        pageno = row.params.page_number
        pageno_html = (f'　ページ番号: <b>{pageno}</b>' if pageno is not None
                       else '　ページ番号: <span style="color:#a00">未読取</span>')
        # cache-busting token so a reload after an edit refetches the previews
        # instead of showing the browser-cached image at the same URL
        import time
        v = int(time.time() * 1000)
        return f"""
<style>
.rbox{{position:absolute;background:transparent;cursor:pointer;
  outline:1px solid transparent}}
.rbox:hover{{background:rgba(221,0,0,.20);outline:2px solid var(--c)}}
button{{padding:3px 8px}}
</style>
<h2>{doc_id} &mdash; page {page_index}</h2>
<p>紙面全体が同一要素なら押す:
  <button onclick="decide('bw')">白黒二値</button>
  <button onclick="decide('gray')">グレースケール</button>
  <button onclick="decide('color')">カラー</button>
  &nbsp;&nbsp;
  <button onclick="backToQueue()">キューに戻る</button>
</p>
<div style="display:flex;gap:16px;flex-wrap:wrap">
  <div>
    <h3>analysis <span style="font-weight:normal;font-size:80%">
      (ボタンを押してからドラッグで領域追加／領域内をダブルクリックで削除)</span></h3>
    <div style="margin:6px 0">
      {mode_buttons}{crop_chip}{pageno_html}
    </div>
    <div id="awrap" style="position:relative;display:inline-block;border:1px solid #ccc">
      <img id="aimg" src="{base}/analysis.png?v={v}" style="max-width:520px;display:block">
      {regions_overlay}
      <div id="rb" style="position:absolute;border:2px dashed #d00;background:rgba(221,0,0,.12);display:none;pointer-events:none"></div>
    </div>
    <p style="font-size:85%;margin:4px 0"><b>検出領域</b>（クラス: {kinds_html}）
      — 各行の「削除」で除去できます（text/figure/photo/table）</p>
    <ul style="font-size:85%;max-height:160px;overflow:auto;max-width:520px">
      {regions_html or '<li>(none)</li>'}</ul>
  </div>
  <div><h3>output <span style="font-weight:normal;font-size:80%">(最終PDF相当)</span></h3>
    <img src="{base}/output.png?v={v}" style="max-width:520px;border:1px solid #ccc">
    <p style="color:#666;font-size:85%;max-width:520px">傾き補正後、<b>page kind</b>と
    領域に応じて二値化／グレー／カラー合成した最終PDF相当の画像です。
    bw ページでは <b>photo 領域</b>だけがグレー/カラーで残り、文字・線画は二値化されます。</p>
  </div>
</div>
<script>
const ORIG_W={ow}, ORIG_H={oh};
(function(){{
  const img=document.getElementById('aimg'),
        wrap=document.getElementById('awrap'), rb=document.getElementById('rb');
  let sx=0, sy=0, drag=false;
  function pos(e){{
    const r=img.getBoundingClientRect();
    return {{x:Math.max(0,Math.min(e.clientX-r.left,r.width)),
             y:Math.max(0,Math.min(e.clientY-r.top,r.height))}};
  }}
  // listen on the wrapper so a drag still starts when the cursor is over an
  // active overlay box (which sits above the image)
  wrap.addEventListener('mousedown',e=>{{
    e.preventDefault(); drag=true; const p=pos(e); sx=p.x; sy=p.y;
    rb.style.left=sx+'px'; rb.style.top=sy+'px';
    rb.style.width='0px'; rb.style.height='0px'; rb.style.display='block';
  }});
  window.addEventListener('mousemove',e=>{{
    if(!drag) return; const p=pos(e);
    rb.style.left=Math.min(sx,p.x)+'px'; rb.style.top=Math.min(sy,p.y)+'px';
    rb.style.width=Math.abs(p.x-sx)+'px'; rb.style.height=Math.abs(p.y-sy)+'px';
  }});
  window.addEventListener('mouseup',e=>{{
    if(!drag) return; drag=false; rb.style.display='none';
    const p=pos(e);
    if(Math.abs(p.x-sx)<4 || Math.abs(p.y-sy)<4) return;   // ignore tiny drags
    const kx=ORIG_W/img.clientWidth, ky=ORIG_H/img.clientHeight;
    const x0=Math.round(Math.min(sx,p.x)*kx), y0=Math.round(Math.min(sy,p.y)*ky),
          x1=Math.round(Math.max(sx,p.x)*kx), y1=Math.round(Math.max(sy,p.y)*ky);
    // drag completes the spec; dispatch by current mode
    if(MODE==='content'){{
      if(confirm('内容領域を設定し、全ページを再計算しますか？'))
        setBox('content',x0,y0,x1,y1);
    }} else if(MODE==='nombre'){{
      if(confirm('ページ番号領域を設定し、全ページを再計算しますか？'))
        setBox('nombre',x0,y0,x1,y1);
    }} else {{
      addPhoto(x0,y0,x1,y1);   // photo: no confirm (cheap, dblclick to undo)
    }}
  }});
}})();
let MODE='gray';
function setMode(m){{
  MODE=m;
  localStorage.setItem('hp_mode', m);   // remember across reloads/pages
  for(const [id,mm] of [['mGray','gray'],['mColor','color'],
                        ['mContent','content'],['mNombre','nombre']]){{
    const b=document.getElementById(id);
    b.style.outline = m===mm ? '2px solid #06c' : 'none';
    b.style.fontWeight = m===mm ? 'bold' : 'normal';
  }}
  // only same-kind boxes are hoverable/deletable for the active mode; others
  // become click-through so an overlapping box can't block them
  const grp = (m==='content'||m==='nombre') ? m : 'photo';
  document.querySelectorAll('.rbox').forEach(el=>{{
    el.style.pointerEvents = (el.dataset.group===grp) ? 'auto' : 'none';
  }});
}}
async function addPhoto(x0,y0,x1,y1){{
  await fetch('/api/page/{doc_id}/{page_index}/region',{{method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{kind:'photo',tone:MODE,x0:x0,y0:y0,x1:x1,y1:y1}})}});
  location.reload();   // box appears on overlay; stays in queue until done
}}
async function setBox(kind,x0,y0,x1,y1){{
  // kind = content | nombre. The endpoint recomputes all pages' crops.
  await fetch('/api/page/{doc_id}/{page_index}/'+kind,{{method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{x0:x0,y0:y0,x1:x1,y1:y1}})}});
  location.reload();
}}
async function delBox(kind){{
  const label = kind==='content' ? '内容領域（削除すると自動検出に戻す）' : 'ページ番号領域';
  if(!confirm(label+'を削除し、全ページを再計算しますか？')) return;
  await fetch('/api/page/{doc_id}/{page_index}/'+kind+'/delete',{{method:'POST'}});
  location.reload();
}}
async function backToQueue(){{
  // recompute every page so the book keeps a uniform page size, then return
  await fetch('/api/doc/{doc_id}/recompute',{{method:'POST'}});
  location.href='/';
}}
async function decide(kind){{
  await fetch('/api/page/{doc_id}/{page_index}/decide',{{method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{page_kind:kind}})}});
  location.href='/';
}}
async function delRegion(i){{
  // double-click deletes immediately (no confirm); re-draw if it was a mistake
  await fetch('/api/page/{doc_id}/{page_index}/region/'+i+'/delete',
    {{method:'POST'}});
  location.reload();
}}
setMode(localStorage.getItem('hp_mode') || 'gray');   // restore last mode
</script>"""

    @app.get("/img/{doc_id}/{page_index}/analysis.png")
    def analysis_png(doc_id: str, page_index: int):
        from ..preview import analysis_overlay

        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        png = analysis_overlay(_load_original(row.params), row.params)
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/img/{doc_id}/{page_index}/output.png")
    def output_png(doc_id: str, page_index: int):
        from ..model import RenderSettings
        from ..preview import output_preview

        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        png = output_preview(_load_original(row.params), row.params, RenderSettings())
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/queue")
    def queue():
        return [
            {"doc_id": r.doc_id, "page_index": r.page_index, "flags": r.flags}
            for r in store.review_queue()
        ]

    @app.get("/api/doc/{doc_id}/pages")
    def list_pages(doc_id: str):
        return [
            {
                "page_index": row.page_index,
                "needs_review": row.review_status == ReviewStatus.NEEDS_REVIEW,
            }
            for row in store.list_pages(doc_id)
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
        elif decision.finish:
            # finalize region edits: leave the queue as corrected
            params.review_status = ReviewStatus.CORRECTED
            params.decided_by = DecidedBy(decision.decided_by)
            store.log_decision(doc_id, page_index, decision.decided_by,
                               "finish", None, True, features)
        store.upsert_page(doc_id, page_index, params)
        return {"status": params.review_status.value}

    @app.post("/api/page/{doc_id}/{page_index}/report")
    def report_page(doc_id: str, page_index: int, report: ReaderReport):
        from ..learn import page_features

        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        params = row.params
        old_status = params.review_status.value
        params.review_status = ReviewStatus.NEEDS_REVIEW
        features = page_features(params)
        store.upsert_page(doc_id, page_index, params)
        store.log_decision(doc_id, page_index, report.decided_by,
                           "reader_report", old_status, "needs_review", features)
        return {"status": "flagged"}

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
        # Do NOT finalize here: the reviewer may add several regions in a row.
        # The page stays in the queue (needs_review) until they click done/
        # approve. The edit is still persisted and logged for learning.
        store.log_decision(doc_id, page_index, edit.decided_by, "region",
                           None, {"kind": edit.kind, "tone": edit.tone,
                                  "box": [edit.x0, edit.y0, edit.x1, edit.y1]},
                           page_features(params))
        store.upsert_page(doc_id, page_index, params)
        return {"status": params.review_status.value,
                "regions": len(params.regions)}

    @app.post("/api/page/{doc_id}/{page_index}/region/{index}/delete")
    def delete_region(doc_id: str, page_index: int, index: int):
        from ..learn import page_features

        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        params = row.params
        if not (0 <= index < len(params.regions)):
            raise HTTPException(404, "region index out of range")
        removed = params.regions.pop(index)
        store.log_decision(
            doc_id, page_index, "human", "region_delete",
            {"kind": removed.kind.value, "source": removed.source,
             "box": [removed.box.x0, removed.box.y0,
                     removed.box.x1, removed.box.y1]},
            None, page_features(params))
        store.upsert_page(doc_id, page_index, params)
        return {"status": params.review_status.value,
                "regions": len(params.regions)}

    def _recompute(doc_id: str) -> int:
        from ..pipeline import recompute_margins
        return recompute_margins(store, doc_id)

    def _ensure_margin(params):
        from ..model import Box, Margin
        if params.margin is None:
            params.margin = Margin(content=Box(0, 0, 1, 1), confidence=1.0)
        return params.margin

    @app.post("/api/page/{doc_id}/{page_index}/content")
    def set_content(doc_id: str, page_index: int, edit: BoxEdit):
        from ..learn import page_features
        from ..model import Box

        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        params = row.params
        m = _ensure_margin(params)
        old = ([m.content.x0, m.content.y0, m.content.x1, m.content.y1]
               if m.content else None)
        m.content = Box(edit.x0, edit.y0, edit.x1, edit.y1)
        m.confidence = 1.0                 # manual = trusted
        store.log_decision(doc_id, page_index, edit.decided_by, "content",
                           old, [edit.x0, edit.y0, edit.x1, edit.y1],
                           page_features(params))
        store.upsert_page(doc_id, page_index, params)
        # crop size depends on the whole parity group -> recompute all pages
        n = _recompute(doc_id)
        return {"status": params.review_status.value, "recomputed": n}

    @app.post("/api/page/{doc_id}/{page_index}/content/delete")
    def delete_content(doc_id: str, page_index: int):
        from ..geometry.margin import find_content_box
        from ..learn import page_features

        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        params = row.params
        # delete = revert content to auto-detection (keeps an existing nombre)
        detected = find_content_box(_load_original(params), params.deskew)
        if params.margin is None:
            params.margin = detected
        else:
            params.margin.content = detected.content
            params.margin.confidence = detected.confidence
        store.log_decision(doc_id, page_index, "human", "content_delete",
                           None, "re-detected", page_features(params))
        store.upsert_page(doc_id, page_index, params)
        n = _recompute(doc_id)
        return {"status": params.review_status.value, "recomputed": n}

    @app.post("/api/page/{doc_id}/{page_index}/nombre")
    def set_nombre(doc_id: str, page_index: int, edit: BoxEdit):
        from ..learn import page_features
        from ..model import Box

        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        params = row.params
        m = _ensure_margin(params)
        old = ([m.nombre_box.x0, m.nombre_box.y0, m.nombre_box.x1, m.nombre_box.y1]
               if m.nombre_box else None)
        m.nombre_box = Box(edit.x0, edit.y0, edit.x1, edit.y1)
        store.log_decision(doc_id, page_index, edit.decided_by, "nombre",
                           old, [edit.x0, edit.y0, edit.x1, edit.y1],
                           page_features(params))
        store.upsert_page(doc_id, page_index, params)
        n = _recompute(doc_id)
        return {"status": params.review_status.value, "recomputed": n}

    @app.post("/api/page/{doc_id}/{page_index}/nombre/delete")
    def delete_nombre(doc_id: str, page_index: int):
        from ..learn import page_features

        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        params = row.params
        if params.margin is None or params.margin.nombre_box is None:
            raise HTTPException(404, "no nombre to delete")
        params.margin.nombre_box = None    # normalize falls back to centering
        store.log_decision(doc_id, page_index, "human", "nombre_delete",
                           None, None, page_features(params))
        store.upsert_page(doc_id, page_index, params)
        n = _recompute(doc_id)
        return {"status": params.review_status.value, "recomputed": n}

    @app.post("/api/doc/{doc_id}/recompute")
    def recompute(doc_id: str):
        return {"recomputed": _recompute(doc_id)}

    return app


def serve(db_path: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    app = create_app(db_path)
    print(f"HokusaiPress review UI -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
