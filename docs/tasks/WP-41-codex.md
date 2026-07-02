# WP-41 (Codex): OPF 書誌出力（Calibre 連携）

計画: `docs/VIEWER_LIBRARY_PLAN.md` の WP-41。担当: Codex。

## ゴール
新規モジュール `src/hokusai_press/export_opf.py` を作り、受け入れテスト
`tests/test_opf.py` を **すべて green** にする。書誌 dict から Dublin Core の
OPF 2.0 パッケージ文書（XML 文字列）を生成する純関数。

## 契約（逸脱不可）
`build_opf(book: dict) -> str`:
1. 整形式の XML を返す（`xml.etree.ElementTree.fromstring` で解析可能）。
   OPF 2.0 の `<package>` 直下に `<metadata>`、その中に Dublin Core 要素
   （名前空間 `http://purl.org/dc/elements/1.1/`）を出す。
2. マッピング:
   - `title` → `dc:title`
   - `author` → `dc:creator`
   - `publisher` → `dc:publisher`
   - `year` → `dc:date`（`str(year)` を含む）
   - `isbn` → `dc:identifier`（値に ISBN 文字列を含む）
3. **欠損フィールドはその DC 要素を出さない**（例外にしない）。
4. **特殊文字（`&`, `<`, `>` 等）は正しくエスケープ**する。手書き文字列連結でなく
   `xml.etree.ElementTree` で組み立て、`ET.tostring(..., encoding="unicode")` で
   文字列化するのが安全（エスケープが自動）。

## 模倣する既存の流儀
- `export_md.py` と同じ「純ロジック・I/O なし・文字列を返す」スタイル。
- `from __future__ import annotations`、簡潔な docstring、標準ライブラリのみ。

## 触ってよいファイル
- `src/hokusai_press/export_opf.py`（新規）
- `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_opf.py`, その他すべて

## 検証（完了条件）
```
python -m pytest tests/test_opf.py -q     # 3 passed
python -m pytest -q                        # 回帰なし
```

## DoD
- 上記2コマンド green（出力貼付）。`export_opf.py` 以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。
