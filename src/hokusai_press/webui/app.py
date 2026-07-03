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


class IngestRequest(BaseModel):
    path: str


class AuthRequest(BaseModel):
    username: str
    password: str


class UserCreateRequest(AuthRequest):
    role: str


def create_app(db_path: str, runner=None, upload_dir=None, require_auth=False):
    import os
    import secrets
    import threading
    import uuid

    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

    app = FastAPI(title="HokusaiPress review")
    store = Store(db_path)
    if upload_dir is None:
        upload_dir = os.path.join(os.path.dirname(db_path) or ".", "uploads")

    if runner is None:
        def runner(path, out_pdf, db_path):
            from ..pipeline import run

            return run(path, out_pdf, db_path)

    jobs = {}
    jobs_lock = threading.Lock()
    sessions: dict[str, str] = {}
    sessions_lock = threading.Lock()

    def _session_user(request: Request):
        token = request.cookies.get("hp_session")
        if not token:
            return None
        with sessions_lock:
            username = sessions.get(token)
        return store.get_user(username) if username else None

    def _start_session(response: Response, username: str) -> None:
        token = secrets.token_urlsafe(32)
        with sessions_lock:
            sessions[token] = username
        response.set_cookie("hp_session", token, httponly=True)

    @app.middleware("http")
    async def require_session(request: Request, call_next):
        if (not require_auth
                or request.url.path in {"/api/login", "/api/setup", "/login"}
                or _session_user(request) is not None):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                {"detail": "authentication required"}, status_code=401
            )
        return RedirectResponse("/login", status_code=302)

    @app.post("/api/setup")
    def setup(auth: AuthRequest, response: Response):
        if store.list_users():
            raise HTTPException(403, "setup already completed")
        store.create_user(auth.username, auth.password, role="admin")
        _start_session(response, auth.username)
        return {"username": auth.username, "role": "admin"}

    @app.post("/api/login")
    def login(auth: AuthRequest, response: Response):
        user = store.verify_user(auth.username, auth.password)
        if user is None:
            raise HTTPException(401, "invalid username or password")
        _start_session(response, user["username"])
        return {"username": user["username"], "role": user["role"]}

    @app.post("/api/logout")
    def logout(request: Request, response: Response):
        token = request.cookies.get("hp_session")
        if token:
            with sessions_lock:
                sessions.pop(token, None)
        response.delete_cookie("hp_session", httponly=True)
        return {"status": "logged_out"}

    @app.get("/api/me")
    def me(request: Request):
        user = _session_user(request)
        if user is None:
            raise HTTPException(401, "authentication required")
        return {"username": user["username"], "role": user["role"]}

    def _require_admin(request: Request):
        user = _session_user(request)
        if user is None or user["role"] != "admin":
            raise HTTPException(403, "admin required")
        return user

    @app.post("/api/users")
    def create_user(user: UserCreateRequest, request: Request):
        _require_admin(request)
        store.create_user(user.username, user.password, role=user.role)
        return {"status": "created"}

    @app.get("/api/users")
    def list_users(request: Request):
        _require_admin(request)
        return [
            {
                "username": user["username"],
                "role": user["role"],
                "created_at": user["created_at"],
            }
            for user in store.list_users()
        ]

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

    def _shell(title: str, body: str, active_path: str = "/") -> str:
        nav_links = [
            ("/library", "📚 書庫"),
            ("/ingest", "📥 取込"),
            ("/", "✏️ レビュー"),
        ]
        nav_html = "".join(
            f'<a href="{path}" class="nav-link {"active" if path == active_path else ""}">{label}</a>'
            for path, label in nav_links
        )
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title} - HokusaiPress</title>
<style>
  :root {{
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --bg: #f8fafc;
    --surface: #ffffff;
    --text: #334155;
    --text-light: #64748b;
    --border: #e2e8f0;
  }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.5;
  }}
  .navbar {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 0 24px; display: flex; align-items: center; height: 60px; gap: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); position: sticky; top: 0; z-index: 100;
  }}
  .navbar-brand {{ font-weight: 700; color: #0f172a; margin-right: 16px; font-size: 1.1rem; }}
  .nav-link {{
    text-decoration: none; color: var(--text-light); font-weight: 500;
    padding: 8px 12px; border-radius: 6px; transition: 0.2s;
  }}
  .nav-link:hover {{ background: #f1f5f9; color: #0f172a; }}
  .nav-link.active {{ background: #eff6ff; color: var(--accent); }}
  .container {{ max-width: 1000px; margin: 32px auto; padding: 0 24px; }}
  .card {{
    background: var(--surface); border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);
    padding: 24px; border: 1px solid #f1f5f9;
  }}
  .btn {{
    display: inline-flex; align-items: center; justify-content: center;
    background: var(--accent); color: white; border: none; padding: 8px 16px;
    border-radius: 6px; font-size: 14px; font-weight: 500; cursor: pointer;
    text-decoration: none; transition: 0.2s;
  }}
  .btn:hover {{ background: var(--accent-hover); }}
  .btn:disabled {{ opacity: 0.6; cursor: not-allowed; }}
  .btn-outline {{
    background: transparent; border: 1px solid var(--border); color: var(--text);
  }}
  .btn-outline:hover {{ background: #f1f5f9; }}
  input[type="text"] {{
    padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px;
    font-size: 14px; outline: none; transition: border-color 0.2s;
  }}
  input[type="text"]:focus {{ border-color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-light); font-weight: 500; font-size: 14px; }}
</style>
</head>
<body>
  <div class="navbar">
    <div class="navbar-brand">HokusaiPress</div>
    {nav_html}
    <div id="userMenu" style="margin-left: auto; display: flex; align-items: center; gap: 12px;"></div>
  </div>
  <div class="container">
    {body}
  </div>
  <script>
  async function _loadUser() {{
      try {{
          const res = await fetch('/api/me');
          if (res.ok) {{
              const data = await res.json();
              document.getElementById('userMenu').innerHTML = `
                  <span style="font-size: 14px; color: var(--text-light)">👤 ${{data.username}}</span>
                  <button class="btn btn-outline" style="padding: 4px 8px; font-size: 13px;" onclick="_doLogout()">ログアウト</button>
              `;
          }}
      }} catch (e) {{}}
  }}
  async function _doLogout() {{
      await fetch('/api/logout', {{method: 'POST'}});
      window.location.href = '/login';
  }}
  _loadUser();
  </script>
</body>
</html>"""

    @app.get("/login", response_class=HTMLResponse)
    def login_view():
        return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Login - HokusaiPress</title>
<style>
  :root {
    --accent: #3b82f6; --accent-hover: #2563eb; --bg: #f8fafc;
    --surface: #ffffff; --text: #334155; --border: #e2e8f0;
  }
  body { margin: 0; background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; }
  .card { background: var(--surface); border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); padding: 32px; width: 100%; max-width: 360px; border: 1px solid #f1f5f9; }
  h2 { margin-top: 0; margin-bottom: 24px; text-align: center; }
  .form-group { margin-bottom: 16px; }
  .form-group label { display: block; margin-bottom: 8px; font-size: 14px; font-weight: 500; }
  input[type="text"], input[type="password"] { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 14px; outline: none; }
  input[type="text"]:focus, input[type="password"]:focus { border-color: var(--accent); }
  .btn { width: 100%; box-sizing: border-box; background: var(--accent); color: white; border: none; padding: 10px; border-radius: 6px; font-size: 14px; font-weight: 500; cursor: pointer; margin-top: 8px; }
  .btn:hover { background: var(--accent-hover); }
  .btn:disabled { opacity: 0.6; cursor: not-allowed; }
  .toggle-link { display: block; text-align: center; margin-top: 16px; font-size: 13px; color: var(--accent); text-decoration: none; cursor: pointer; }
  .toggle-link:hover { text-decoration: underline; }
  #errorMsg { color: #dc2626; font-size: 14px; margin-bottom: 16px; display: none; text-align: center; }
</style>
</head>
<body>
  <div class="card">
    <h2 id="formTitle">ログイン</h2>
    <div id="errorMsg"></div>
    <div class="form-group">
      <label>ユーザー名</label>
      <input type="text" id="username" onkeydown="if(event.key==='Enter') submitForm()">
    </div>
    <div class="form-group">
      <label>パスワード</label>
      <input type="password" id="password" onkeydown="if(event.key==='Enter') submitForm()">
    </div>
    <button class="btn" id="submitBtn" onclick="submitForm()">ログイン</button>
    <a class="toggle-link" id="toggleMode" onclick="toggleMode()">初回セットアップの方はこちら</a>
  </div>
<script>
let mode = 'login';
function toggleMode() {
    mode = mode === 'login' ? 'setup' : 'login';
    document.getElementById('formTitle').innerText = mode === 'login' ? 'ログイン' : '初期管理者を作成';
    document.getElementById('submitBtn').innerText = mode === 'login' ? 'ログイン' : '作成';
    document.getElementById('toggleMode').innerText = mode === 'login' ? '初回セットアップの方はこちら' : 'ログイン画面に戻る';
    document.getElementById('errorMsg').style.display = 'none';
}
async function submitForm() {
    const u = document.getElementById('username').value.trim();
    const p = document.getElementById('password').value;
    if (!u || !p) return;
    
    const btn = document.getElementById('submitBtn');
    const err = document.getElementById('errorMsg');
    btn.disabled = true;
    err.style.display = 'none';
    
    try {
        const res = await fetch(mode === 'login' ? '/api/login' : '/api/setup', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username: u, password: p})
        });
        
        if (res.ok) {
            window.location.href = '/library';
        } else {
            const status = res.status;
            if (mode === 'setup' && status === 403) {
                err.innerText = '既にセットアップ済みです。ログインしてください。';
                toggleMode();
            } else {
                err.innerText = mode === 'login' ? 'ログインに失敗しました' : 'エラーが発生しました';
            }
            err.style.display = 'block';
        }
    } catch(e) {
        err.innerText = '通信エラーが発生しました';
        err.style.display = 'block';
    } finally {
        btn.disabled = false;
    }
}
</script>
</body>
</html>"""

    @app.get("/admin/users", response_class=HTMLResponse)
    def admin_users_view():
        body = """
<style>
.form-card { margin-bottom: 24px; padding: 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
.form-group { margin-bottom: 12px; }
.form-group label { display: block; font-size: 13px; margin-bottom: 4px; font-weight: 500; }
.form-group input, .form-group select { padding: 8px; width: 200px; border: 1px solid var(--border); border-radius: 4px; box-sizing: border-box; }
#errorMsg { color: #dc2626; font-size: 14px; margin-bottom: 12px; display: none; }
</style>
<div class="card">
  <h2>ユーザー管理</h2>
  <div id="adminContent">読み込み中...</div>
</div>
<script>
async function loadUsers() {
    const res = await fetch('/api/users');
    const content = document.getElementById('adminContent');
    if (res.status === 403) {
        content.innerHTML = '<p>管理者のみアクセスできます</p>';
        return;
    }
    const users = await res.json();
    let html = `
    <div class="form-card">
      <h3 style="margin-top:0;font-size:15px">新規ユーザー追加</h3>
      <div id="errorMsg"></div>
      <div style="display:flex; gap:16px; align-items:flex-end; flex-wrap:wrap;">
        <div class="form-group"><label>ユーザー名</label><input type="text" id="newUsername"></div>
        <div class="form-group"><label>パスワード</label><input type="password" id="newPassword"></div>
        <div class="form-group"><label>ロール</label>
          <select id="newRole"><option value="user">user</option><option value="admin">admin</option></select>
        </div>
        <button class="btn" style="margin-bottom:12px; width:auto; padding:8px 16px;" onclick="addUser()">追加</button>
      </div>
    </div>
    <table>
      <thead><tr><th>ユーザー名</th><th>ロール</th><th>作成日時</th></tr></thead>
      <tbody>
    `;
    users.forEach(u => {
        html += `<tr><td>${escapeHtml(u.username)}</td><td>${escapeHtml(u.role)}</td><td>${escapeHtml(u.created_at)}</td></tr>`;
    });
    html += '</tbody></table>';
    content.innerHTML = html;
}
async function addUser() {
    const u = document.getElementById('newUsername').value.trim();
    const p = document.getElementById('newPassword').value;
    const r = document.getElementById('newRole').value;
    if (!u || !p) return;
    const err = document.getElementById('errorMsg');
    err.style.display = 'none';
    
    try {
        const res = await fetch('/api/users', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username: u, password: p, role: r})
        });
        if (res.ok) {
            loadUsers();
        } else {
            err.innerText = 'エラーが発生しました';
            err.style.display = 'block';
        }
    } catch (e) {
        err.innerText = '通信エラーが発生しました';
        err.style.display = 'block';
    }
}
function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/[&<>"']/g, function(m) {
        return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[m];
    });
}
loadUsers();
</script>
"""
        return _shell("ユーザー管理", body, active_path="")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _shell("レビューキュー", """
<div class="card">
  <h2>レビューキュー</h2>
  <div id="queue-container">読み込み中...</div>
</div>
<script>
async function loadQueue() {
    const res = await fetch('/api/queue');
    const rows = await res.json();
    const container = document.getElementById('queue-container');
    if (rows.length === 0) {
        container.innerHTML = '<div style="text-align:center;padding:40px;color:var(--text-light)">レビュー待ちはありません 🎉</div>';
        return;
    }
    let html = '<table><thead><tr><th>ページ</th><th>Doc ID</th><th>Flags</th><th>類似ページ数 (Rank Score)</th></tr></thead><tbody>';
    rows.forEach(r => {
        html += `<tr>
            <td><a href="/page/${r.doc_id}/${r.page_index}" class="btn btn-outline" style="padding:4px 8px">確認する</a></td>
            <td>${r.doc_id} p.${r.page_index}</td>
            <td>${r.flags.join(', ')}</td>
            <td>${r.rank_score.toFixed(2)}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}
loadQueue();
</script>
""", active_path="/")

    @app.get("/page/{doc_id}/{page_index}", response_class=HTMLResponse)
    def page_view(doc_id: str, page_index: int, request: Request):
        user = _session_user(request)
        owner = store.get_doc_owner(doc_id)
        if user is not None and owner is not None and owner != user["username"]:
            raise HTTPException(403)
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

    @app.get("/read/{doc_id}", response_class=HTMLResponse)
    def read_view(doc_id: str, request: Request):
        user = _session_user(request)
        owner = store.get_doc_owner(doc_id)
        if user is not None and owner is not None and owner != user["username"]:
            raise HTTPException(403)
        body = f"""
<style>
  .pages-wrapper {{ display: flex; flex-direction: column; align-items: center; }}
  .page-container {{
    position: relative;
    margin: 20px 0;
    background: #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    min-height: 600px;
    width: 800px;
    max-width: 100%;
  }}
  .page-container img {{
    display: block;
    width: 100%;
    height: auto;
  }}
  .marker {{
    position: absolute;
    left: 0;
    top: 20px;
    width: 6px;
    height: 40px;
    background: #e00;
  }}
  .marker.reported {{
    background: #0a0;
  }}
  .controls {{
    position: absolute;
    right: -140px;
    top: 20px;
    width: 120px;
  }}
  .report-btn {{
    padding: 6px 12px;
    cursor: pointer;
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 13px;
    color: var(--text);
  }}
  .report-btn:hover {{ background: #f1f5f9; }}
  .feedback {{
    color: #16a34a;
    font-size: 13px;
    opacity: 0;
    transition: opacity 1s;
    margin-top: 6px;
  }}
  .feedback.show {{
    opacity: 1;
    transition: opacity 0.1s;
  }}
  .read-header {{ text-align: center; margin-bottom: 24px; }}
</style>
<div class="read-header">
  <h2>{doc_id}</h2>
</div>
<div class="pages-wrapper" id="pages"></div>
<script>
const DOC_ID = "{doc_id}";
let currentGen = 0;
const MAX_CONCURRENT = 3;
let activeLoads = 0;
let loadQueue = [];

async function init() {{
    currentGen++;
    const myGen = currentGen;
    
    const res = await fetch(`/api/doc/${{DOC_ID}}/pages`);
    const pages = await res.json();
    
    if (myGen !== currentGen) return; // 世代管理
    
    const container = document.getElementById('pages');
    
    pages.forEach(p => {{
        const div = document.createElement('div');
        div.className = 'page-container';
        div.dataset.pageIndex = p.page_index;
        
        if (p.needs_review) {{
            const marker = document.createElement('div');
            marker.className = 'marker';
            marker.id = 'marker-' + p.page_index;
            div.appendChild(marker);
        }}
        
        const controls = document.createElement('div');
        controls.className = 'controls';
        
        const btn = document.createElement('button');
        btn.className = 'report-btn';
        btn.innerText = 'このページを報告';
        btn.onclick = () => reportPage(p.page_index, btn);
        controls.appendChild(btn);
        
        const fb = document.createElement('div');
        fb.className = 'feedback';
        fb.id = 'fb-' + p.page_index;
        fb.innerText = '報告しました';
        controls.appendChild(fb);
        
        div.appendChild(controls);
        container.appendChild(div);
        
        observer.observe(div);
    }});
}}

async function reportPage(pageIndex, btn) {{
    btn.disabled = true;
    try {{
        const res = await fetch(`/api/page/${{DOC_ID}}/${{pageIndex}}/report`, {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{decided_by: 'human'}})
        }});
        if (res.ok) {{
            const marker = document.getElementById('marker-' + pageIndex);
            if (marker) marker.classList.add('reported');
            
            const fb = document.getElementById('fb-' + pageIndex);
            fb.classList.add('show');
            setTimeout(() => fb.classList.remove('show'), 1000); // 1秒でフェード
        }}
    }} finally {{
        btn.disabled = false;
    }}
}}

function processQueue() {{
    if (activeLoads >= MAX_CONCURRENT || loadQueue.length === 0) return;
    
    const taskIndex = loadQueue.findIndex(t => t.isIntersecting);
    if (taskIndex === -1) return; // viewport内になければロードしない
    
    const task = loadQueue.splice(taskIndex, 1)[0];
    activeLoads++;
    
    const img = document.createElement('img');
    const myGen = currentGen;
    
    img.onload = img.onerror = () => {{
        if (myGen !== currentGen) return; // 古い世代のコールバックは破棄
        activeLoads--;
        processQueue();
    }};
    
    img.src = `/img/${{DOC_ID}}/${{task.pageIndex}}/output.png`;
    task.div.appendChild(img);
    
    // 他の待機タスクも再帰的にチェック
    processQueue();
}}

const observer = new IntersectionObserver((entries) => {{
    let changed = false;
    entries.forEach(entry => {{
        const div = entry.target;
        const pageIndex = div.dataset.pageIndex;
        
        let task = loadQueue.find(t => t.pageIndex === pageIndex);
        
        if (entry.isIntersecting) {{
            if (!div.querySelector('img') && !task) {{
                loadQueue.push({{div, pageIndex, isIntersecting: true}});
                changed = true;
            }} else if (task) {{
                task.isIntersecting = true;
                changed = true;
            }}
        }} else {{
            if (task) {{
                task.isIntersecting = false;
            }}
        }}
    }});
    if (changed) processQueue();
}}, {{ rootMargin: '100% 0px' }});

init();
</script>
"""
        return _shell(f"Read {doc_id}", body, active_path="")

    @app.get("/library", response_class=HTMLResponse)
    def library_view():
        body = """
<style>
.search-bar { display: flex; gap: 8px; margin-bottom: 24px; }
.search-bar input { flex: 1; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }
.book-card { display: flex; flex-direction: column; gap: 12px; }
.book-title { font-weight: bold; font-size: 16px; }
.book-meta { color: var(--text-light); font-size: 14px; }
.hit-card { margin-bottom: 16px; }
.snippet { margin-top: 8px; font-size: 14px; color: var(--text); background: var(--bg); padding: 12px; border-radius: 6px; border: 1px solid var(--border); white-space: pre-wrap; }
mark { background-color: #fef08a; padding: 0 4px; border-radius: 2px; }
</style>
<div class="card" style="margin-bottom: 24px;">
  <div class="search-bar">
    <input type="text" id="searchInput" placeholder="検索..." onkeydown="if(event.key==='Enter') doSearch()">
    <button class="btn" onclick="doSearch()">検索</button>
  </div>
</div>
<div id="results"></div>
<script>
async function init() {
    const res = await fetch('/api/library');
    const docs = await res.json();
    const results = document.getElementById('results');
    
    let html = '<h2>書籍一覧</h2><div class="grid">';
    docs.forEach(doc => {
        const title = doc.title ? doc.title : doc.doc_id;
        html += `<div class="card book-card">
            <div class="book-title">${escapeHtml(title)}</div>
            <div class="book-meta">${doc.page_count} ページ</div>
            <div style="margin-top:auto"><a href="/read/${doc.doc_id}" class="btn" style="width:100%;box-sizing:border-box">📖 読む</a></div>
        </div>`;
    });
    html += '</div>';
    results.innerHTML = html;
}

async function doSearch() {
    const q = document.getElementById('searchInput').value.trim();
    if (!q) { init(); return; }
    
    const res = await fetch('/api/search?q=' + encodeURIComponent(q));
    const hits = await res.json();
    const results = document.getElementById('results');
    
    let html = `<h2>"${escapeHtml(q)}" の検索結果</h2>`;
    if (hits.length === 0) {
        html += '<p style="color:var(--text-light)">見つかりませんでした。</p>';
    } else {
        hits.forEach(hit => {
            const safeSnippet = escapeHtml(hit.snippet).replace(/\\[\\.\\.\\.\\]/g, '<mark>[...]</mark>');
            html += `<div class="card hit-card">
                <div style="margin-bottom:8px"><a href="/read/${hit.doc_id}" style="color:var(--accent);text-decoration:none;font-weight:bold">${escapeHtml(hit.doc_id)} - ページ ${hit.page_index}</a></div>
                <div class="snippet">${safeSnippet}</div>
            </div>`;
        });
    }
    results.innerHTML = html;
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/[&<>"']/g, function(m) {
        return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[m];
    });
}
init();
</script>
"""
        return _shell("書庫", body, active_path="/library")

    @app.get("/ingest", response_class=HTMLResponse)
    def ingest_view():
        body = """
<style>
.ingest-form { display: flex; gap: 8px; margin-bottom: 24px; }
.ingest-form input { flex: 1; }
.drop-zone { border: 2px dashed var(--border); border-radius: 8px; padding: 32px; text-align: center; cursor: pointer; margin-bottom: 24px; transition: all 0.2s; background: var(--bg); }
.drop-zone.dragover { border-color: var(--accent); background: #eff6ff; }
.drop-zone-icon { font-size: 32px; margin-bottom: 8px; }
.drop-zone-text { color: var(--text-light); font-size: 14px; }
.upload-progress { margin-top: 12px; text-align: left; font-size: 14px; }
.upload-item { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid var(--border); }
.job-card { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; padding: 16px; }
.job-status { font-weight: bold; display: flex; align-items: center; gap: 8px; width: 120px; }
.status-running { color: #d97706; }
.status-done { color: #16a34a; }
.status-error { color: #dc2626; }
.spinner { animation: spin 1s linear infinite; display: inline-block; }
@keyframes spin { 100% { transform: rotate(360deg); } }
.job-info { flex: 1; }
.job-error { color: #dc2626; font-size: 14px; margin-top: 4px; }
</style>
<div class="card">
  <h2>新規取込</h2>
  <div id="dropZone" class="drop-zone" onclick="document.getElementById('fileInput').click()">
    <div class="drop-zone-icon">📄</div>
    <div class="drop-zone-text">ここにファイルをドラッグ、またはクリックして選択</div>
    <input type="file" id="fileInput" style="display:none" multiple accept=".pdf">
    <div id="uploadProgress" class="upload-progress"></div>
  </div>
  <h3 style="font-size:14px;color:var(--text-light);margin-bottom:12px;margin-top:0">またはサーバ上のパスを指定</h3>
  <div class="ingest-form">
    <input type="text" id="pathInput" placeholder="サーバ上のファイルパス (例: /path/to/book.pdf)" onkeydown="if(event.key==='Enter') submitIngest()">
    <button class="btn" id="submitBtn" onclick="submitIngest()">取込開始</button>
  </div>
  <div id="errorMsg" style="color:#dc2626;margin-bottom:16px;display:none"></div>
</div>
<h3 style="margin-top:32px">ジョブ一覧</h3>
<div id="jobsList">読み込み中...</div>
<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); }, false);
});
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});
['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});
dropZone.addEventListener('drop', e => handleFiles(e.dataTransfer.files), false);
fileInput.addEventListener('change', e => handleFiles(e.target.files), false);

async function handleFiles(files) {
    if (!files || files.length === 0) return;
    const errMsg = document.getElementById('errorMsg');
    errMsg.style.display = 'none';
    uploadProgress.innerHTML = '';
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const item = document.createElement('div');
        item.className = 'upload-item';
        item.innerHTML = `<span>${escapeHtml(file.name)}</span><span id="up-status-${i}">⏳ アップロード中...</span>`;
        uploadProgress.appendChild(item);
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                document.getElementById(`up-status-${i}`).innerHTML = '❌ エラー';
                document.getElementById(`up-status-${i}`).style.color = '#dc2626';
                errMsg.textContent = data.detail || `アップロードエラー: ${file.name}`;
                errMsg.style.display = 'block';
            } else {
                document.getElementById(`up-status-${i}`).innerHTML = '✅ 完了';
                document.getElementById(`up-status-${i}`).style.color = '#16a34a';
                pollJobs();
            }
        } catch (e) {
            document.getElementById(`up-status-${i}`).innerHTML = '❌ エラー';
            document.getElementById(`up-status-${i}`).style.color = '#dc2626';
            errMsg.textContent = e.toString();
            errMsg.style.display = 'block';
        }
    }
}

async function submitIngest() {
    const path = document.getElementById('pathInput').value.trim();
    if (!path) return;
    const btn = document.getElementById('submitBtn');
    const errMsg = document.getElementById('errorMsg');
    btn.disabled = true;
    errMsg.style.display = 'none';
    
    try {
        const res = await fetch('/api/ingest', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path: path})
        });
        if (!res.ok) {
            const data = await res.json();
            errMsg.textContent = data.detail || 'エラーが発生しました';
            errMsg.style.display = 'block';
        } else {
            document.getElementById('pathInput').value = '';
            pollJobs();
        }
    } catch(e) {
        errMsg.textContent = e.toString();
        errMsg.style.display = 'block';
    } finally {
        btn.disabled = false;
    }
}

async function pollJobs() {
    try {
        const res = await fetch('/api/jobs');
        const jobs = await res.json();
        const container = document.getElementById('jobsList');
        if (jobs.length === 0) {
            container.innerHTML = '<p style="color:var(--text-light)">ジョブはありません</p>';
            return;
        }
        
        let html = '';
        jobs.forEach(job => {
            let statusHtml = '';
            let actionHtml = '';
            if (job.status === 'running') {
                statusHtml = '<span class="status-running"><span class="spinner">⏳</span> 処理中</span>';
            } else if (job.status === 'done') {
                statusHtml = '<span class="status-done">✅ 完了</span>';
                actionHtml = `<a href="/read/${job.doc_id}" class="btn btn-outline" style="padding:4px 8px">📖 読む</a>`;
            } else if (job.status === 'error') {
                statusHtml = '<span class="status-error">❌ エラー</span>';
            }
            
            html += `<div class="card job-card">
                <div class="job-status">${statusHtml}</div>
                <div class="job-info">
                    <div style="font-weight:bold">${escapeHtml(job.doc_id)}</div>
                    ${job.error ? `<div class="job-error">${escapeHtml(job.error)}</div>` : ''}
                </div>
                <div>${actionHtml}</div>
            </div>`;
        });
        container.innerHTML = html;
    } catch(e) {
        console.error(e);
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/[&<>"']/g, function(m) {
        return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[m];
    });
}

pollJobs();
setInterval(pollJobs, 3000);
</script>
"""
        return _shell("取込", body, active_path="/ingest")

    @app.get("/img/{doc_id}/{page_index}/analysis.png")
    def analysis_png(doc_id: str, page_index: int, request: Request):
        from ..preview import analysis_overlay

        user = _session_user(request)
        owner = store.get_doc_owner(doc_id)
        if user is not None and owner is not None and owner != user["username"]:
            raise HTTPException(403)
        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        png = analysis_overlay(_load_original(row.params), row.params)
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/img/{doc_id}/{page_index}/output.png")
    def output_png(doc_id: str, page_index: int, request: Request):
        from ..model import RenderSettings
        from ..preview import output_preview

        user = _session_user(request)
        owner = store.get_doc_owner(doc_id)
        if user is not None and owner is not None and owner != user["username"]:
            raise HTTPException(403)
        row = store.get_page(doc_id, page_index)
        if row is None:
            raise HTTPException(404, "page not found")
        png = output_preview(_load_original(row.params), row.params, RenderSettings())
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/queue")
    def queue(request: Request):
        from .. import queue_rank

        user = _session_user(request)
        pending = {
            (r.doc_id, r.page_index): r
            for r in store.review_queue()
        }
        return [
            {
                "doc_id": doc_id,
                "page_index": page_index,
                "flags": pending[(doc_id, page_index)].flags,
                "rank_score": score,
            }
            for doc_id in store.doc_ids()
            if (user is None or store.get_doc_owner(doc_id) is None
                or store.get_doc_owner(doc_id) == user["username"])
            for page_index, score in queue_rank.rank_pending(
                store, doc_id, similar_threshold=0.5
            )
            if (doc_id, page_index) in pending
        ]

    def _start_ingest_job(path: str, username: str | None = None) -> dict:
        doc_id = os.path.basename(path)
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "doc_id": doc_id,
            "status": "running",
            "error": None,
        }
        with jobs_lock:
            jobs[job_id] = job

        def execute():
            try:
                os.makedirs("out", exist_ok=True)
                runner(path, os.path.join("out", doc_id), db_path)
            except Exception as exc:
                with jobs_lock:
                    job["status"] = "error"
                    job["error"] = str(exc)
            else:
                if username is not None:
                    store.set_doc_owner(doc_id, username)
                with jobs_lock:
                    job["status"] = "done"

        threading.Thread(target=execute, daemon=True).start()
        return {"job_id": job_id, "doc_id": doc_id}

    @app.post("/api/ingest")
    def ingest(body: IngestRequest, request: Request):
        if not os.path.isfile(body.path):
            raise HTTPException(400, "file not found")
        user = _session_user(request)
        return _start_ingest_job(
            body.path, user["username"] if user is not None else None
        )

    @app.post("/api/upload")
    def upload(request: Request, file: UploadFile = File(...)):
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, file.filename)
        with open(path, "wb") as destination:
            destination.write(file.file.read())
        user = _session_user(request)
        return _start_ingest_job(
            path, user["username"] if user is not None else None
        )

    @app.get("/api/jobs")
    def list_jobs():
        with jobs_lock:
            return [dict(job) for job in reversed(jobs.values())]

    @app.get("/api/library")
    def library(request: Request):
        user = _session_user(request)
        owned_doc_ids = (
            set(store.doc_ids_for_user(user["username"])) if user is not None else set()
        )
        rows = []
        for doc_id in store.doc_ids():
            if (user is not None and doc_id not in owned_doc_ids
                    and store.get_doc_owner(doc_id) is not None):
                continue
            book = store.get_book(doc_id)
            rows.append({
                "doc_id": doc_id,
                "page_count": len(store.list_pages(doc_id)),
                "title": book["title"] if book else None,
            })
        return rows

    @app.get("/api/search")
    def search(request: Request, q: str = "", limit: int = 50):
        if q == "":
            return []
        user = _session_user(request)
        return [
            {
                "doc_id": hit.doc_id,
                "page_index": hit.page_index,
                "snippet": hit.snippet,
            }
            for hit in store.search(q, limit)
            if (user is None or store.get_doc_owner(hit.doc_id) is None
                or store.get_doc_owner(hit.doc_id) == user["username"])
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


def serve(db_path: str, host: str = "127.0.0.1", port: int = 8765,
          require_auth: bool = True) -> None:
    import uvicorn

    app = create_app(db_path, require_auth=require_auth)
    print(f"HokusaiPress review UI -> http://{host}:{port}"
          + (" (login required)" if require_auth else " (no auth -- local use only)"))
    uvicorn.run(app, host=host, port=port)
