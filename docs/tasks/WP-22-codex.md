# WP-22 (Codex): 修正の波及エンジン

計画: `docs/VIEWER_LIBRARY_PLAN.md` の WP-22。担当: Codex。前提: WP-21 完了。

## ゴール
新規モジュール `src/hokusai_press/propagate.py` を作り、受け入れテスト
`tests/test_propagate.py` を **すべて green** にする。1件の page_kind 修正を、
同一 doc 内の構造が似たページへ広げる。

## 契約（逸脱不可）
`propagate(store, doc_id, source_index, page_kind, *, auto_threshold,
queue_threshold, decided_by="propagation") -> dict`:
1. `store.list_pages(doc_id)` で全ページを取得。`source_index` のページを
   基準に、他ページを `similarity.rank_similar`（WP-21）で距離付けする。
2. 各対象ページについて:
   - **距離 <= auto_threshold**: `params.page_kind = page_kind` にして
     `store.upsert_page` で保存し、`store.log_decision(doc_id, i, decided_by,
     field="page_kind", old_value=<元kind>, new_value=<新kind>, features=
     learn.page_features(params))` を記録。`applied` に page_index を追加。
   - **auto_threshold < 距離 <= queue_threshold**: 変更せず
     `params.review_status = ReviewStatus.NEEDS_REVIEW` にして保存＋decision記録。
     `queued` に追加。
   - **距離 > queue_threshold**: 何もしない。
3. **人手決定を上書きしない**: `params.decided_by == DecidedBy.HUMAN` の
   ページは applied にも queued にも入れず、一切変更しない。
4. 基準ページ自身（`source_index`）は結果に含めない。
5. 戻り値は `{"applied": [page_index...], "queued": [page_index...]}`。
   自動変更は必ず decision に残す（監査・取り消し可能性のため）。

## 模倣する既存の流儀
- store 書き込みは既存メソッド（`upsert_page`/`log_decision`）をそのまま使う。
- 特徴量は `learn.page_features`、類似度は `similarity.rank_similar` を再利用。
- `from __future__ import annotations`、簡潔な docstring。

## 触ってよいファイル
- `src/hokusai_press/propagate.py`（新規）
- `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_propagate.py`, `store.py`, `model.py`, `similarity.py`,
  `learn.py`, その他すべて

## 検証（完了条件）
```
python -m pytest tests/test_propagate.py -q     # 5 passed
python -m pytest -q                              # 回帰なし
```

## DoD
- 上記2コマンド green（出力貼付）。`propagate.py` 以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。
