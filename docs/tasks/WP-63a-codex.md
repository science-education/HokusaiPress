# WP-63a (Codex): アップロード API（D&D 取込のバックエンド）

前提: WP-61a（`POST /api/ingest`, `GET /api/jobs`）完了済み。担当: Codex。
`python-multipart` は既に `pyproject.toml` の `web` extra に追加・インストール済み。

## ゴール
受け入れテスト `tests/test_webui_upload.py` を **すべて green** にする。
`src/hokusai_press/webui/app.py` に `POST /api/upload` を追加する。

## 契約（逸脱不可）
1. `create_app(db_path, runner=None, upload_dir=None)` — 第3引数を追加。
   - `upload_dir` 省略時は `os.path.join(os.path.dirname(db_path) or ".", "uploads")`。
   - 既存呼び出し（`create_app(db)`, `create_app(db, runner=...)`）は無改修で動くこと。
2. `POST /api/upload`（`file: UploadFile = File(...)`、FastAPI の標準パターン）:
   - `os.makedirs(upload_dir, exist_ok=True)`。
   - 保存先ファイル名は **元のファイル名をそのまま使う**
     （`os.path.join(upload_dir, file.filename)`）。同名なら上書きでよい
     （簡易実装。衝突対策は将来課題）。
   - アップロードされたバイト列をそのまま書き込む（`file.file.read()` で全読み）。
   - 保存後は **`/api/ingest` と同じジョブ起動ロジックを再利用**する
     （重複実装しない。`ingest` の中身をヘルパー関数に切り出し、両エンドポイントから
     呼ぶ形にする。例: `def _start_ingest_job(path: str) -> dict: ...` を
     `create_app` 内に定義し、`ingest()` と `upload()` の両方から呼ぶ）。
   - 戻り値は `{"job_id": str, "doc_id": str}`（`ingest` と同じ形）。
3. `doc_id` は `os.path.basename(保存パス)` そのまま（日本語ファイル名もそのまま使う）。

## 模倣する既存の流儀
- 既存の `POST /api/ingest`（`ingest()` 関数）のジョブ生成・スレッド起動・
  `jobs`/`jobs_lock` の使い方をそのまま再利用する。

## 触ってよいファイル
- `src/hokusai_press/webui/app.py`

## 触ってはならないファイル（契約）
- `tests/test_webui_upload.py`, その他すべて
- 既存ルートの挙動（`/api/ingest` の外部インターフェースは変えない）

## 検証（完了条件）
```
python -m pytest tests/test_webui_upload.py -q     # 3 passed
python -m pytest -q                                  # 回帰なし
```

## DoD
- 上記2コマンド green（出力貼付）。`app.py` 以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。
