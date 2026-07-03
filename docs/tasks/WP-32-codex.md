# WP-32 (Codex): レビューキューに能動学習の順位を反映

計画: `docs/VIEWER_LIBRARY_PLAN.md` の WP-32。担当: Codex。前提: WP-31 完了。

## ゴール
`GET /api/queue`（`src/hokusai_press/webui/app.py`）が返す各ページの並び順を、
`page_index` の生順ではなく `queue_rank.rank_pending` の順位（＝その判断が解決する
類似ページ数が多い順）にする。受け入れテスト `tests/test_webui_queue_ranked.py`
を **すべて green** にする。

## 契約（逸脱不可）
1. `queue()` エンドポイントの実装を変更する:
   - `doc_id` ごとに `queue_rank.rank_pending(store, doc_id,
     similar_threshold=0.5)` を呼び、そのページ順を使う
     （`similar_threshold=0.5` を既定値として使う。将来クエリパラメータ化しても
     良いが今回は固定でよい）。
   - 複数 doc がある場合は doc の出現順（`store.doc_ids()` の順）で連結する。
   - 各要素の dict に **`"rank_score": float`**（`rank_pending` が返すスコア）を
     追加する。既存キー（`doc_id`, `page_index`, `flags`）は維持する。
2. 既存の `needs_review` フィルタ（`store.review_queue` 相当の絞り込み）は
   維持する。

## 模倣する既存の流儀
- `queue_rank.rank_pending` をそのまま使う（再実装しない）。
- 既存 `queue()` の dict 構築スタイル（`{"doc_id": ..., "page_index": ..., "flags": ...}`）
  にキーを1つ足すだけ。

## 触ってよいファイル
- `src/hokusai_press/webui/app.py`（`queue()` の変更のみ）
- `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_webui_queue_ranked.py`, `queue_rank.py`, `store.py`, その他すべて
- `app.py` 内の `queue()` 以外のルート

## 検証（完了条件）
```
python -m pytest tests/test_webui_queue_ranked.py -q     # 2 passed
python -m pytest -q                                        # 回帰なし
```

## DoD
- 上記2コマンド green（出力貼付）。`app.py` 以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。
