# WP-11 (Antigravity): 読書モード UI（連続ページ表示）

計画: `docs/VIEWER_LIBRARY_PLAN.md` の WP-11。担当: Antigravity。
前提: WP-11a（バックエンド API）が先に完了していること。

> Antigravity 向けの注意: 下の「作るもの」以外の設計判断をしないこと。データ取得先・
> 挙動・受け入れ条件は下に確定済み。迷ったら勝手に拡張せず、そのまま実装する。

## 作るもの（これだけ）
`src/hokusai_press/webui/app.py` の `create_app` に、読書用の1ルートを追加する:
`GET /read/{doc_id}` → HTML を返す。既存の `page_view`（`/page/{doc_id}/{page_index}`）
と **同じ流儀**（Python 文字列で HTML を組み立て、`HTMLResponse` で返す）で書く。
テンプレートエンジンやビルド工程は導入しない（依存を増やさない）。

このページは対象 doc の全ページを縦スクロールの連続画像として表示する:
1. 読み込み時に `GET /api/doc/{doc_id}/pages`（WP-11a）で全ページ一覧を取得。
2. 各ページを `GET /img/{doc_id}/{page_index}/output.png`（既存）で画像表示。
   画像は **遅延読み込み**（`loading="lazy"` もしくは IntersectionObserver）。
3. `needs_review: true` のページには欄外に控えめなマーカー（例: 左端の細い帯や小さな印）。
   ページ内容は隠さない。
4. 各ページに「このページを報告」ボタン。押すと
   `POST /api/page/{doc_id}/{page_index}/report`（body `{"decided_by":"human"}`）を送り、
   成功したらそのページのマーカーを「報告済み」表示に更新する（再読込不要）。

## pdf2clipboard から移植する挙動（VIEWER_LIBRARY_PLAN §0 の技術資産）
画像は PDF.js でなくサーバ生成 PNG なので、以下だけ取り込む:
- **世代管理**: 速いスクロールで前の doc の非同期結果が現在の表示を上書きしないよう、
  読み込み世代カウンタで古い結果を破棄する。
- **レンダ同時実行数の制御**: 画像の同時読み込み数に上限を設け、ビューポート外はロードしない。
- **軽いフィードバック**: 報告ボタンの確認表示は約1秒でフェード、内容を隠す半透明オーバーレイは使わない。
（中央基準ズーム等の凝った UX は WP-11 の範囲外。今回は縦スクロール＋遅延読み込みで十分。）

## 触ってよいファイル
- `src/hokusai_press/webui/app.py`（`/read/{doc_id}` の追加のみ。既存ルートは変更しない）

## 触ってはならないファイル（契約。変更が必要なら止めて報告）
- `store.py`, `model.py`, `learn.py`, `tests/**`, その他すべて
- 既存の webui ルート（`/`, `/page/...`, `/api/...`, `/img/...`）のロジック

## 検証（完了条件 — 証拠を貼付）
1. 既存テストが回帰しないこと: `python -m pytest -q` が全て green。
2. 起動して目視:
   ```
   python -m hokusai_press.webui --db hokusai.db   # もしくは webui.app.serve(...)
   ```
   ブラウザで `/read/<doc_id>` を開き、(a) 全ページが順に表示される、
   (b) フラグ付きページにマーカーが出る、(c) 報告ボタンで `/report` が呼ばれ表示が更新される、
   (d) 高速スクロールで表示が壊れない、をスクリーンショットで示す。

## DoD
- `python -m pytest -q` が green（回帰なし。UI 自体のテストは必須ではないが、既存を壊さない）。
- 目視4項目のスクリーンショット。
- `docs/HANDOFF.md` 先頭に3行追記（何を・どのルートに・確認方法）。
