# HokusaiPress

スキャン PDF を **美しく整い・データは軽く・全文検索できる** 電子書籍 PDF に変換する
バッチ処理アプリ。判定が難しいページだけを人間（または AI）のレビューに回し、その
判断を蓄積して自動判定を継続的に改善します。

## 設計思想（ScanTailor から継承）

**元画像は最後まで不可逆化しない。** 各処理段は画像を書き換えず *パラメータ*
（傾き角・余白枠・領域クラス・判断）だけを記録し、最終レンダラのみが元画像を
読み、**1回のリサンプリング**で出力画素を生成します。レビュー UI での修正も
「パラメータの上書き」であり、画像の再加工ではありません。したがってどのページも
いつでも無劣化で再生成できます。

- 中間生成物は画像でなく [`PageParams`](src/hokusai_press/model.py)（傾き・余白・
  領域・ページ種別・確信度・フラグ）
- 300dpi ラスタライズは OCR/幾何解析の**作業解像度**。最終画像は元画像（スキャン
  PDF なら埋め込み原画像を pikepdf で無劣化抽出）から計算
- deskew + 切り抜き + 解像度統一を1つのアフィン変換に合成し、元画像を一度だけワープ

## パイプライン

```
入力(PDF/画像)
  source.py     原画像抽出 + OCR用300dpiラスタ（倍率を記録）
  geometry/     deskew（確信度つき）/ 内容ボックス・ノンブル基準マージン
  content.py    字/図/写真の峻別（DBNet行+認識 → RT-DETR図 → 残差写真）
  pipeline.py   パラメータ確定 + 低確信ページにフラグ → SQLite 保存
  render.py     元画像から1パスでワープ → 検索可能PDF（G4/JBIG2 + 透明文字層）
  webui/        フラグ付きページのみレビュー → 上書き → 判断ログ → 再生成
```

## 使い方

```bash
pip install -e .[ocr,web]
# Mac（Apple Silicon）で NDL-OCR/Yomitoku と GPU (MPS) 高速化を利用する場合
pip install -e ".[mac-ocr,web]"
# pyjbig2 は通常インストールで自動導入されます
hokusai-press run scan.pdf --out book.pdf --device npu      # バッチ解析+生成 (Intel NPU)
hokusai-press run scan.pdf --out book.pdf --device mps      # バッチ解析+生成 (Mac GPU 高速化)
hokusai-press run scan.pdf --out book.pdf --layout-engine none --text-engine ppocr-v6 --runtime onnxruntime --device npu
hokusai-press run scan.pdf --out book.pdf --layout-engine paddle-vl --text-engine ppocr-v6 --device npu
hokusai-press queue                                          # 要レビュー一覧
hokusai-press review                                         # レビューWeb UI
hokusai-press rebuild scan.pdf --doc scan.pdf --out book.pdf  # 修正後に再生成
hokusai-press learn --out model.json                        # 判断ログから学習
hokusai-press run scan.pdf --out book.pdf --learned-model model.json
```

`--no-ocr` で OCR を省略し幾何処理＋ヒューリスティック分離のみでも動作します。
OCR/レイアウトは `--layout-engine` と `--text-engine` で別々に選択できます。
既定は既存互換の `yomitoku`（レイアウト）+ `ndlocr`（文字OCR）です。Paddle 系では
`pp-structurev3` / `paddle-vl` をレイアウトに、`ppocr-v6` を文字OCRに選択できます。
旧 `--ocr-engine` はまとめ指定の互換オプションとして残しています。
Intel NPU で実機確認済みの Paddle 経路は `--text-engine ppocr-v6 --runtime onnxruntime
--device npu` です。`pp-structurev3` と `paddle-vl` は依存・backend 制約が大きいため
現時点では実験扱いです。

レビュー UI はフラグ付きページのみを表示し、各ページの「解析オーバーレイ
（傾き補正後＋内容枠＋領域分類＋ノンブル）」と「出力プレビュー」を並べて表示。
ページ種別（bw/gray/color）を上書きすると判断が特徴量つきでログされ、`rebuild`
が修正済みパラメータから PDF を無劣化再生成します（出力はパラメータの純関数）。

### 種別は2軸 — 「bw ページの中のグレー画像」

処理は2軸で、混同しないことが重要です。

- **page kind**（`bw`/`gray`/`color`）= ページ**全体**のベース層コーデック。
  `gray`/`color` は「全面が写真・図版」「地紙ごと退色」など**全面トーン維持**が
  要るときの**フォールバック**。
- **region kind**（`text`/`figure`/`photo`）= **領域ごと**のトーン処理（本命）。
  `text`・`figure`(線画) は二値（くっきり）、`photo`(連続調) はその矩形だけ
  グレー/カラー JPEG として上に重ねます（MRC）。

したがって **bw ページ内のグレー写真は、page kind を bw のままにし、その範囲を
`photo` 領域として追加**します。文字は二値で鮮明・写真だけグレーで軽い、が同一
ページ内で両立します。`photo` 領域のグレー/カラーは既定で領域の彩度から自動判定
しますが、レビュー UI の **tone**（`auto`/`gray`/`color`）で固定もできます
（`Region.tone`、MRC のオーバーレイ符号化に反映）。

## ライセンス

- 本リポジトリのコード: **GPL-3.0-or-later**。精密傾き補正・ノンブル基準マージンは
  [ScanTailorAdvancedTATEGAKI](https://github.com/science-education/ScanTailorAdvancedTATEGAKI)
  （GPLv3）由来のロジックを移植するため、継承して GPLv3 とします。
- OCR エンジン [hybrid-ocr](https://github.com/science-education/Yomitoku_NDL-OCR-Lite)
  を使う場合、検出器（YomiToku DBNet, CC BY-NC-SA 4.0）の制約により
  **生成された PDF は非商用利用に限定**されます。商用利用が必要な場合は検出器を
  差し替え可能な構成にしてあります（content.py の OCR インターフェース）。
- JBIG2エンコーダ [pyjbig2](https://github.com/science-education/pyjbig2)
  はApache-2.0です。`jbig2enc 0.32`由来のlossless generic-region方式で、
  シンボル辞書を使わないため文字置換リスクのある非可逆JBIG2ではありません。

## 状態

実装済み・検証済み（29テスト、CI ubuntu+windows）:
- 非破壊パラメータモデル + JSON/SQLite 永続化
- deskew（確信度つき、縦横書き両対応）
- ノンブル基準マージン正規化（パリティ別・均一判型・クリップなし・ノンブル縦アンカー）
- 1パスレンダラ + 物理ページサイズ（実書籍で 4.70×7.27 inch 均一を確認）
- 多層 MRC（二値テキスト層 G4/JBIG2 + 写真層 JPEG、字は鮮明・写真は軽量）
- 字/図/写真の峻別: DBNet 行+認識、RT-DETRv2 図領域フック（layout_provider）、
  残差インク写真ヒューリスティック
- バッチ → 低確信ページのみレビュー（解析オーバーレイ+出力プレビュー、種別/領域編集）
  → 判断ログ → rebuild（パラメータ純関数で無劣化再生成）
- 判断ログからの page-kind 学習（最近傍重心、依存追加なし）→ 自動判定へ反映
- 検索可能テキスト層

次段: ノンブル基準マージンの綴じ方向対応（spine 整列）の本格 C++ 移植、
RT-DETRv2 アダプタの実機検証、レビュー UI の領域ドラッグ描画。
