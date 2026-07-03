# WP-73 (Codex): セッション認証（ログイン/初期セットアップ/ルート保護）

家庭内サーバのマルチユーザー化・第3段。前提: WP-71（`user`テーブル/`Store.create_user`
等）完了済み。担当: Codex。

## ゴール
受け入れテスト `tests/test_auth.py` を **すべて green** にする。
`src/hokusai_press/webui/app.py` を拡張する。**既存の280件超のテストは
`create_app(db)` を認証なしで呼ぶ前提のまま**なので、後方互換性を壊さないこと。

## 契約（逸脱不可）
1. `create_app(db_path, runner=None, upload_dir=None, require_auth=False)`
   — 第4引数を追加。**既定は `False`**（既存呼び出しは無改修で動く）。
2. `require_auth=True` のときだけ有効になる **HTTP ミドルウェア**
   （`@app.middleware("http")`）を追加する。個々のルート関数は変更しない
   （シグネチャに `Depends` を足して回らない — 保護はミドルウェア1箇所に集約）。
   - 公開パス許可リスト（認証不要）: `/api/login`, `/api/setup`。
   - それ以外のパスで有効なセッションが無ければ:
     - パスが `/api/` で始まる → `401` の JSON `{"detail": "authentication required"}`
     - それ以外（HTML ページ）→ `/login` へ `302` リダイレクト
       （`/login` ページ自体は別 WP で作る。今回は redirect を返すだけでよい）。
3. セッション: アプリ内のメモリ dict（`sessions: dict[str, str]` token→username）を
   `threading.Lock` で保護（既存の `jobs`/`jobs_lock` と同じパターン）。
   - Cookie 名 `hp_session`、`httponly=True`。トークンは `secrets.token_urlsafe(32)`。
4. `POST /api/setup`（body `{"username","password"}`）:
   - `store.list_users()` が空でなければ `403`。
   - 空なら `store.create_user(username, password, role="admin")`、
     セッション作成＋Cookie 設定、`{"username","role"}` を返す（`200`）。
5. `POST /api/login`（body `{"username","password"}`）:
   - `store.verify_user` が成功すればセッション作成＋Cookie 設定、
     `{"username","role"}` を返す（`200`）。失敗なら `401`。
6. `POST /api/logout`: Cookie を無効化しセッションを破棄。未ログインでも `200`。
7. `GET /api/me`: 有効なセッションがあれば `{"username","role"}`（`200`）、
   無ければ `401`。
8. 管理者専用ユーザー管理（**`require_auth` の値に関係なく常にこの2ルート自身が
   セッション必須**。ミドルウェアとは独立にルート内でチェックする）:
   - `POST /api/users`（body `{"username","password","role"}`）:
     現在のセッションが無い/`role != "admin"` なら `403`。
     OK なら `store.create_user(...)` して `{"status":"created"}`。
   - `GET /api/users`: 同様に admin のみ。`store.list_users()` から
     **`password_hash`/`salt` を除いた** `{"username","role","created_at"}` の
     配列を返す。

## 模倣する既存の流儀
- Cookie/セッション管理は既存 `jobs`/`jobs_lock` パターン（クロージャ内 dict +
  Lock）を踏襲。
- Pydantic body モデルを `Decision` 等と同じ書き方で定義。

## 触ってよいファイル
- `src/hokusai_press/webui/app.py`
- `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_auth.py`, `store.py`, その他すべて
- 既存ルートの中身（保護はミドルウェアで一括、個々のルート関数は書き換えない）

## 検証（完了条件）
```
python -m pytest tests/test_auth.py -q     # 9 passed
python -m pytest -q                         # 回帰なし（既存 require_auth=False 経路は無傷）
```

## DoD
- 上記2コマンド green（出力貼付）。`app.py` 以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。
