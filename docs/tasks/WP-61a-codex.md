# WP-61a (Codex): 取込 API（登録→パイプライン実行→進捗）

計画: 登録→修整→閲覧ループの「登録」バックエンド。担当: Codex。

## ゴール
受け入れテスト `tests/test_webui_ingest.py` を **すべて green** にする。
`src/hokusai_press/webui/app.py` に取込 API を追加する。

## 契約（逸脱不可）
1. `create_app(db_path: str, runner=None)` に省略可能な第2引数を追加。
   - `runner(path, out_pdf, db_path)` はバッチ処理関数。`None` のときの既定は
     `pipeline.run` を呼ぶ薄いラッパ（**関数内で遅延 import**。モジュール先頭で
     pipeline を import しない — webui はモデル依存を持たないこと）。
     既定ラッパの `out_pdf` は `out/<doc_id>` とし `os.makedirs("out", exist_ok=True)`。
   - 既存の `create_app(db)` 呼び出しは全てそのまま動くこと（後方互換）。
2. `POST /api/ingest`（body `{"path": str}`）:
   - `os.path.isfile(path)` でなければ `HTTPException(400)`。
   - 有効なら `doc_id = os.path.basename(path)`、ジョブを作り
     `threading.Thread(daemon=True)` で `runner(path, out_pdf, db_path)` を実行。
     成功で status `"done"`、例外なら `"error"`（`error` に `str(例外)` を保存）。
   - 返り値 `{"job_id": str, "doc_id": str}`。
3. `GET /api/jobs` → 新しい順の配列
   `[{"job_id": str, "doc_id": str, "status": "running"|"done"|"error", "error": str|None}]`。
4. ジョブ一覧はアプリ内のメモリ dict でよい（プロセス再起動で消えてよい）。
   スレッドから更新するので `threading.Lock` で守る。

## 模倣する既存の流儀
- ルートは `create_app` クロージャ内に既存と同じ書き方で追加。
- Pydantic body モデル（`Decision` 等と同様）で `path` を受ける。

## 触ってよいファイル
- `src/hokusai_press/webui/app.py`
- `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_webui_ingest.py`, `pipeline.py`, `store.py`, その他すべて
- 既存ルートのロジック

## 検証（完了条件）
```
python -m pytest tests/test_webui_ingest.py -q     # 4 passed
python -m pytest -q                                 # 回帰なし
```

## DoD
- 上記2コマンド green（出力貼付）。`app.py` 以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。
