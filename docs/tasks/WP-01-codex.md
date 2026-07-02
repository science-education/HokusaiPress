# WP-01 (Codex): FTS5 全文検索を store.py に追加

計画: `docs/VIEWER_LIBRARY_PLAN.md` の WP-01。担当: Codex。

## ゴール（何を作るか）
`Store` に、保存済みページの OCR テキストに対する全文検索を追加する。
受け入れテスト `tests/test_search.py` を **すべて green** にすること。設計判断は
そのテストと本書で確定済み。テストを弱めて通すことは禁止。実装で満たす。

## 契約（テストから読み取れる要求 — 逸脱不可）
1. `hokusai_press.store` に dataclass `SearchHit(doc_id: str, page_index: int, snippet: str)` を追加。
2. `Store.search(query: str, limit: int = 50) -> list[SearchHit]`:
   - 3文字以上のクエリ → FTS5 `MATCH` を使い、`bm25()` の昇順（関連度順）で返す。
     `snippet` は FTS5 の `snippet()` 出力で、一致語を含むこと。
   - 1〜2文字のクエリ → trigram は3文字未満に一致しないため、`text LIKE '%query%'` で
     フォールバック検索する。`snippet` はテキストの一部でよい（一致語を含めること）。
   - 一致なし → `[]`。
3. インデックスは既存の `pages` と同じ `hokusai.db` 内の FTS5 仮想テーブル。
   `tokenize='trigram'` を使う（日本語を分かち書きせず部分一致するため）。
   列は `doc_id UNINDEXED, page_index UNINDEXED, text`。
4. **自動同期**: `upsert_page()` は保存と同時に、そのページの `params.reading_text()` を
   インデックスへ反映する（該当 `(doc_id, page_index)` を DELETE してから、テキストが
   空でなければ INSERT）。再 upsert で古い語が残らないこと（テスト
   `test_reupsert_removes_stale_hits`）。テキストが空のページは索引に載せない
   （`test_blank_page_has_no_text_indexed`）。
5. `Store.reindex_all() -> None`: FTS テーブルを空にし、`pages` 全行から
   `reading_text()` を読んで索引を作り直す（既存 db / スキーマ移行用）。
6. FTS テーブルは `_SCHEMA` の `executescript` で `CREATE VIRTUAL TABLE IF NOT EXISTS`
   として作成し、既存 db を開いても壊れないこと。

## 模倣する既存の流儀（ゼロから様式を発明しない）
- `store.py` の他メソッドと同様に、全ての DB アクセスを `with self._lock:` で囲む。
- コミットは書き込みメソッドの末尾で `self.conn.commit()`。
- dataclass は `PageRow` の書き方に合わせる。`from __future__ import annotations` は維持。
- OCR テキストの取得は `PageParams.reading_text()`（既存メソッド）をそのまま使う。

## 触ってよいファイル
- `src/hokusai_press/store.py`（このWPの実装先）
- `docs/HANDOFF.md`（DoD の追記のみ）

## 触ってはならないファイル（契約。変更が必要なら実装を止めて報告）
- `tests/test_search.py`（受け入れテスト。読むだけ）
- `src/hokusai_press/model.py`, `src/hokusai_press/pipeline.py`, その他すべて

## 検証（この2つが完了条件）
```
python -m pytest tests/test_search.py -q          # 8 passed
python -m pytest -q                                # 既存テストも全て green（回帰なし）
```

## DoD
- 上記2コマンドが green（テスト出力を貼付して報告）。
- `docs/HANDOFF.md` 先頭に日付見出しで3行追記（何を・どこに・テスト名）。
- `store.py` 以外の実装ファイルを変更していないこと。
