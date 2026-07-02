# WP-13 (Antigravity): 書庫画面（一覧＋全文検索）

計画: `docs/VIEWER_LIBRARY_PLAN.md` の WP-13。担当: Antigravity。
前提: WP-13a（`/api/library`, `/api/search`）と WP-11（`/read/{doc_id}`）完了済み。

> Antigravity 向け注意: 下の「作るもの」以外の設計判断をしないこと。データ取得先・
> 挙動は確定済み。勝手に機能を足さず、そのまま実装する。

## 作るもの（これだけ）
`src/hokusai_press/webui/app.py` の `create_app` に、書庫トップの1ルートを追加:
`GET /library` → HTML を返す。既存 `page_view`/`read_view` と **同じ流儀**
（Python 文字列で HTML 組み立て → `HTMLResponse`）。ビルド工程やテンプレートエンジンは
導入しない。

このページは2部構成:
1. **蔵書一覧**: 読み込み時に `GET /api/library` を取得し、各 doc をカード/行で表示。
   `title`（無ければ `doc_id`）と `page_count` を出す。各項目は
   `/read/{doc_id}`（WP-11 の読書画面）へのリンクにする。
2. **全文検索**: 検索ボックス1つ。入力して実行すると `GET /api/search?q=<入力>` を叩き、
   ヒットを一覧表示。各ヒットは `doc_id`・`page_index`・`snippet` を示し、
   `/read/{doc_id}`（可能なら該当ページへスクロールするアンカー等でも良いが必須ではない）
   へのリンクにする。空入力なら検索しない。

## 実装メモ（迷わないために）
- 検索は Enter もしくはボタンで発火。デバウンスや自動補完は不要。
- `snippet` はサーバが `[` `]` で一致箇所を囲んだテキスト（そのまま表示でよい。
  HTML エスケープは行うこと）。
- スタイルは既存 UI と大きく乖離させない。凝ったデザインは不要。

## 触ってよいファイル
- `src/hokusai_press/webui/app.py`（`/library` の追加のみ。既存ルートは変更しない）

## 触ってはならないファイル（契約。必要なら止めて報告）
- `store.py`, `model.py`, `cli.py`, `tests/**`, その他すべて
- 既存の webui ルート（`/`, `/read/...`, `/page/...`, `/api/...`, `/img/...`）

## 検証（完了条件 — 証拠を貼付）
1. 回帰なし: `.venv/bin/python -m pytest -q` が全て green。
2. 目視: アプリ起動 → `/library` を開き、(a) 蔵書一覧が出てリンクで `/read/...` に飛べる、
   (b) 検索語を入れるとヒットが出て `snippet` が表示される、(c) 空検索で何も起きない、
   をスクリーンショットで示す。

## DoD
- `.venv/bin/python -m pytest -q` green（回帰なし）。
- 目視3項目のスクリーンショット。
- `docs/HANDOFF.md` 先頭に3行追記。
