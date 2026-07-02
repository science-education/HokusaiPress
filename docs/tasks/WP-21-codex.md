# WP-21 (Codex): ページ類似度（波及の基盤）

計画: `docs/VIEWER_LIBRARY_PLAN.md` の WP-21。担当: Codex。

## ゴール
新規モジュール `src/hokusai_press/similarity.py` を作り、受け入れテスト
`tests/test_similarity.py` を **すべて green** にする。純ロジック（numpy のみ、
画像 I/O なし。DB 依存なし）。`learn.py`/`profile.py` と同じ「依存を増やさない
純関数モジュール」の流儀で書く。

## 契約（逸脱不可）
1. `page_vector(params: PageParams) -> np.ndarray`:
   - `learn.page_features(params)`（既存）の値を並べた固定長ベクトルを返す。
     `learn.FEATURES` の順で `float` 配列にする。NaN を含めない。
2. `rank_similar(target: PageParams, candidates: list[tuple[K, PageParams]],
   k: int | None = None) -> list[tuple[K, float]]`:
   - プール（target + 全 candidate）の各次元について平均・標準偏差を出し、
     z-score 正規化する。**標準偏差が 0 の次元はその次元の寄与を 0 にする**
     （0 除算・NaN を出さない）。
   - 各 candidate と target の正規化ベクトル間 **ユークリッド距離** を計算し、
     距離の昇順で `(key, distance)` を返す。
   - `k` が指定されれば先頭 `k` 件。`candidates` が空なら `[]`。
   - 同一特徴の candidate は距離 `0.0`。

## 模倣する既存の流儀
- `learn.py` の numpy 使用・`FEATURES` 定義・正規化の考え方を踏襲。
- `from __future__ import annotations` を付ける。docstring は簡潔に目的を書く。

## 触ってよいファイル
- `src/hokusai_press/similarity.py`（新規）
- `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_similarity.py`, `learn.py`, `model.py`, その他すべて

## 検証（完了条件）
```
python -m pytest tests/test_similarity.py -q     # 5 passed
python -m pytest -q                               # 回帰なし
```

## DoD
- 上記2コマンド green（出力貼付）。`similarity.py` 以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。
