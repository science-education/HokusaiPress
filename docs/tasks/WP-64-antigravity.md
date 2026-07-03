# WP-64 (Antigravity): レビューUIの調整パネルとフラグの平易表示

担当: Antigravity。前提: 以下の API は全て実装済み。
- `GET /api/flags` → `{flag値: {label, desc, check}}`（日本語の平易な説明）
- `POST /api/page/{doc}/{i}/rotate` body `{"delta": 90|180|270, "decided_by": "human"}`
- `POST /api/page/{doc}/{i}/deskew` body `{"angle_deg": float(±15以内), "decided_by": "human"}`
- 既存: `POST /api/page/{doc}/{i}/content`（本文枠=マージンの調整、BoxEdit）

> Antigravity 向け注意: 新しいエンドポイントを追加しない。設計判断をしない。
> 外部ライブラリ・CDN・ビルド工程を導入しない。既存 JS/HTML の流儀に合わせる。

## 作るもの（3点）

### 1. レビューキュー(`/`)のフラグを平易表示
`index()` の HTML/JS を変更: 読み込み時に `GET /api/flags` を1回取得し、
各行の flags を生の英語値でなく **`label`のバッジ**（角丸チップ）で表示。
バッジにマウスを載せると `desc` がツールチップ（`title`属性で可）。

### 2. ページ詳細(`/page/{doc}/{i}`)に「このページの問題点」パネル
画像の上（目立つ位置）にカードを1つ追加:
- そのページの flags を1件ずつ「**label** — desc / 👉 check」形式でリスト表示。
  データは `GET /api/flags` と、既存のページ情報（`GET /api/page/{doc}/{i}` が
  flags を返す。返さない場合はページHTMLに埋め込まれている flags を使う。
  どちらも無ければ page_view の Python 側で flags を JSON としてテンプレートに
  埋め込んでよい — 表示目的のみの変更は許可）。
- flags が空なら パネル自体を出さない。

### 3. ページ詳細に「調整」パネル（回転・傾き・本文枠）
同ページに操作カードを1つ追加:
- **回転**: 「↺90°」「↻90°」「180°」の3ボタン。↻90°=delta 90、↺90°=delta 270、
  180°=delta 180 を `/rotate` に POST。成功したら `location.reload()`（画像・
  枠は全てサーバ側で追従済みのため再読込が最も確実）。
- **傾き**: 現在角度を表示し、`-2°〜+2°` を 0.05° 刻みの `<input type="range">`
  ＋数値入力で指定、「適用」ボタンで `/deskew` に POST → reload。
  微調整ボタン「-0.1」「+0.1」も付ける。
- **本文枠(マージン)**: 既存のドラッグ指定に加え、「上下左右を数値で微調整」:
  現在の content box を表示し、各辺 ±px ボタン（±5px）で更新して既存の
  `/content` エンドポイントに POST → reload。現在の content box の値は
  page_view の Python 側でテンプレートに埋め込んでよい。
- 各操作の失敗時は alert かインラインエラーで表示。

## 触ってよいファイル
- `src/hokusai_press/webui/app.py`（`index()` と `page_view()` の HTML/JS、
  および表示目的のデータ埋め込みのみ。API ルートのロジックは変更しない）

## 触ってはならないもの（契約）
- すべての `/api/...` ルートのロジック、`tests/**`、他ファイルすべて

## 検証（完了条件 — 証拠を貼付）
1. `.venv/bin/python -m pytest -q` 全て green。
2. 起動して確認: (a) `/` でフラグが日本語バッジ表示、(b) `/page/...` に問題点
   パネルが出て説明が読める、(c) 回転ボタンで画像が回り枠が追従する、
   (d) 傾きスライダで角度が変わる、(e) 本文枠の±ボタンで枠が動く。

## DoD
- pytest green。`app.py` 以外は無改修。
- `docs/HANDOFF.md` 先頭に3行追記。
