# WP-02 (Codex): 書誌メタデータの読み取り API と `search` CLI

計画: `docs/VIEWER_LIBRARY_PLAN.md` の WP-02。担当: Codex。前提: WP-01 完了済み。

## ゴール
受け入れテスト `tests/test_library.py` を **すべて green** にする。書き込み側
（`Store.save_book`）は実装済み。今回は読み取り側と CLI を足す。

## 契約（逸脱不可）
1. `Store.list_books() -> list[dict]`: `book` テーブル全行を `title` 昇順で返す
   （各行は `dict(row)`）。
2. `Store.get_book(book_id: str) -> dict | None`: 無ければ `None`。
3. `Store.search_books(query: str) -> list[dict]`: `title`/`author`/`publisher` の
   いずれかに `query` を含む行（大文字小文字無視の部分一致、`LIKE '%q%'`）を
   `title` 昇順で返す。
4. CLI サブコマンド `search`:
   - `hokusai-press search <query> [--db hokusai.db] [--limit N]`
   - `Store.search(query, limit)`（WP-01）を呼び、各ヒットを1行で標準出力へ:
     `doc_id` と `page_index` を必ず含める（例: `book.pdf:4  …snippet…`）。
   - `main([...])` は `0` を返す。ヒット0件でも `0`。

## 模倣する既存の流儀
- store の読み取りは `page_features`（`dict(r)` を返す）の書き方に合わせ、
  全アクセスを `with self._lock:` で囲む。
- CLI は `cli.py` の既存サブコマンド（`export` など）の登録と、`main` 内の
  `if args.command == "...":` ブロック＋`store = Store(args.db)` の流儀に従う。

## 触ってよいファイル
- `src/hokusai_press/store.py`
- `src/hokusai_press/cli.py`
- `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_library.py`（受け入れテスト。読むだけ）
- `model.py`, `webui/**`, その他すべて

## 検証（完了条件）
```
python -m pytest tests/test_library.py -q     # 4 passed
python -m pytest -q                            # 回帰なし
```

## DoD
- 上記2コマンド green（出力貼付）。指定2ファイル以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。
