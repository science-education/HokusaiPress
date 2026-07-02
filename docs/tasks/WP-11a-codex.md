# WP-11a (Codex): 読書モードのバックエンドAPI

計画: `docs/VIEWER_LIBRARY_PLAN.md` の WP-11 のバックエンド分。担当: Codex。
これは WP-11（Antigravity のフロントエンド）が消費する API。先にこちらを green にする。

## ゴール
受け入れテスト `tests/test_webui_reading.py` を **すべて green** にする。
`src/hokusai_press/webui/app.py` に2つのエンドポイントを追加する。

## 契約（逸脱不可）
1. `GET /api/doc/{doc_id}/pages` → レビュー待ちだけでなく **全ページ** を
   `page_index` 昇順で返す。各要素は少なくとも
   `{"page_index": int, "needs_review": bool}`。
   （`Store.list_pages(doc_id)` を使う。`needs_review` は
   `row.review_status == "needs_review"`。）
2. `POST /api/page/{doc_id}/{page_index}/report`（body: `{"decided_by": str}`）:
   - 対象ページが無ければ `404`。
   - あれば `review_status` を `needs_review` にして `upsert_page` で保存し、
     `store.log_decision(doc_id, page_index, decided_by, field="reader_report",
     old_value=<元のstatus>, new_value="needs_review", features=<page_features>)`
     を1件記録する。`{"status": "flagged"}` を返す。
   - 教師信号なので `field` は必ず `"reader_report"`。

## 模倣する既存の流儀
- 既存の `decide` / `set_content` エンドポイント（同 app.py 内）の書き方に合わせる:
  Pydantic body モデル、`store.get_page` で存在確認、`HTTPException(404)`、
  変更後 `store.upsert_page`、決定は `store.log_decision`。
- 特徴量ベクトルは、他エンドポイントが decisions に積む際に使っている
  `learn.page_features(params)` を再利用する（同じ流儀で）。
- `create_app(db_path)` のクロージャ内にルートを足す既存パターンに従う。

## 触ってよいファイル
- `src/hokusai_press/webui/app.py`
- `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_webui_reading.py`（受け入れテスト。読むだけ）
- `store.py`, `model.py`, `learn.py`, その他すべて

## 検証（完了条件）
```
python -m pytest tests/test_webui_reading.py -q     # 3 passed
python -m pytest -q                                  # 回帰なし（全て green）
```

## DoD
- 上記2コマンド green（出力貼付）。`app.py` 以外の実装ファイル未変更。
- `docs/HANDOFF.md` 先頭に3行追記。
