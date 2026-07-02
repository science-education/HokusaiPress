# WP-13a (Codex): 書庫・検索の HTTP エンドポイント

計画: `docs/VIEWER_LIBRARY_PLAN.md` の WP-13 バックエンド。担当: Codex。
前提: WP-01（`Store.search`）, WP-02（`Store.list_books`/`get_book`）完了済み。

## ゴール
受け入れテスト `tests/test_webui_library.py` を **すべて green** にする。
`src/hokusai_press/webui/app.py` に2エンドポイントを追加。

## 契約（逸脱不可）
1. `GET /api/library` → doc ごとに1要素の配列。各要素は
   `{"doc_id": str, "page_count": int, "title": str|None}`。
   - doc は `store.doc_ids()`、ページ数は `len(store.list_pages(doc_id))`。
   - `title` は `store.get_book(doc_id)` があればその `title`、無ければ `None`。
2. `GET /api/search?q=<query>&limit=<N>` → `store.search(q, limit)` の結果を
   `[{"doc_id", "page_index", "snippet"}]` で返す。`limit` 省略時は 50。
   - `q` が空文字なら `[]` を返す（検索を呼ばない）。

## 模倣する既存の流儀
- `create_app(db_path)` クロージャ内に、既存の `@app.get("/api/queue")` などと
  同じ書き方でルートを追加。JSON はそのまま dict/list を返す（FastAPI が変換）。
- クエリパラメータは `q: str = ""`, `limit: int = 50` のように関数引数で受ける。

## 触ってよいファイル
- `src/hokusai_press/webui/app.py`
- `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_webui_library.py`, `store.py`, `model.py`, その他すべて
- 既存の webui ルートのロジック

## 検証（完了条件）
```
python -m pytest tests/test_webui_library.py -q     # 3 passed
python -m pytest -q                                  # 回帰なし
```

## DoD
- 上記2コマンド green（出力貼付）。`app.py` 以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。
