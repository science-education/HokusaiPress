# WP-31 (Codex): 能動学習キューの順位付け

計画: `docs/VIEWER_LIBRARY_PLAN.md` の WP-31。担当: Codex。前提: WP-21 完了。

## ゴール
新規モジュール `src/hokusai_press/queue_rank.py` を作り、受け入れテスト
`tests/test_queue_rank.py` を **すべて green** にする。レビュー待ちページを
「その判断が解決する類似ページ数」の多い順に並べる（1回の人手回答の価値を最大化）。

## 契約（逸脱不可）
`rank_pending(store, doc_id, *, similar_threshold, k=None)
-> list[tuple[int, float]]`:
1. 対象は `store.list_pages(doc_id)` のうち
   `review_status == ReviewStatus.NEEDS_REVIEW.value` のページ（pending）のみ。
2. 各 pending ページ p のスコア = **他の pending ページのうち、p との
   `similarity` 距離が `similar_threshold` 以下であるものの個数**（クラスタサイズ）。
   距離は WP-21 の `similarity.rank_similar` を使って算出する。
3. スコアの **降順** に `(page_index, score)` を返す。同点は `page_index` 昇順。
4. `k` 指定時は先頭 `k` 件。pending が無ければ `[]`。

## 模倣する既存の流儀
- store 読み取りは既存メソッド（`list_pages`）を使う。類似度は
  `similarity.rank_similar`/`page_vector` を再利用（自前で距離を再実装しない）。
- `from __future__ import annotations`、numpy 可、簡潔な docstring。

## 触ってよいファイル
- `src/hokusai_press/queue_rank.py`（新規）
- `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_queue_rank.py`, `store.py`, `model.py`, `similarity.py`, その他すべて

## 検証（完了条件）
```
python -m pytest tests/test_queue_rank.py -q     # 4 passed
python -m pytest -q                               # 回帰なし
```

## DoD
- 上記2コマンド green（出力貼付）。`queue_rank.py` 以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。
