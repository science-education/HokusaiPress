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
hokusai-press run scan.pdf --out book.pdf --device npu   # バッチ
hokusai-press queue                                       # 要レビュー一覧
hokusai-press review                                      # レビューWeb UI
```

`--no-ocr` で OCR を省略し幾何処理＋ヒューリスティック分離のみでも動作します。

## ライセンス

- 本リポジトリのコード: **GPL-3.0-or-later**。精密傾き補正・ノンブル基準マージンは
  [ScanTailorAdvancedTATEGAKI](https://github.com/science-education/ScanTailorAdvancedTATEGAKI)
  （GPLv3）由来のロジックを移植するため、継承して GPLv3 とします。
- OCR エンジン [hybrid-ocr](https://github.com/science-education/Yomitoku_NDL-OCR-Lite)
  を使う場合、検出器（YomiToku DBNet, CC BY-NC-SA 4.0）の制約により
  **生成された PDF は非商用利用に限定**されます。商用利用が必要な場合は検出器を
  差し替え可能な構成にしてあります（content.py の OCR インターフェース）。

## 状態

v0 骨格。データモデル・ジョブ/判断ストア・deskew（確信度つき）・1パスレンダラ・
バッチCLI・レビューUIの骨組みが稼働。ノンブル基準マージンのページ間最適化（C++
からの移植）と多層 MRC は次段。
