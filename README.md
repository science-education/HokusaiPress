# HokusaiPress

スキャン PDF を **美しく整い・データは軽く・全文検索できる** 電子書籍 PDF に変換する
バッチ処理アプリ。判定が難しいページだけを人間（または AI）のレビューに回し、その
判断を蓄積して自動判定を継続的に改善します。

## 設計思想（ScanTailor から継承）

**入力原本は書き換えない。** 解析時の影除去などはメモリ上の作業画像にだけ適用し、
永続化するのは *パラメータ*（傾き角・余白枠・領域クラス・判断）です。最終レンダラは
毎回入力原本から出力を生成するため、中間画像の再圧縮・再加工が累積しません。
レビュー UI での修正も「パラメータの上書き」であり、同じ原本から再生成できます。

- 中間生成物は画像でなく [`PageParams`](src/hokusai_press/model.py)（傾き・余白・
  領域・ページ種別・確信度・フラグ）
- OCR/幾何解析には作業画像を使用（既知DPIでは最大300dpi、DPI不明では長辺最大
  3000px）。最終画像は元画像（単一画像のスキャンPDFなら埋め込み画像を
  pypdfium2で抽出）から計算
- deskew + 切り抜き + 解像度統一を1つのアフィン変換に合成し、元画像を一度だけワープ

## パイプライン

```
入力(PDF/画像)
  source.py     原画像抽出 + OCR用作業画像（倍率を記録）
  geometry/     deskew（確信度つき）/ 内容ボックス・ノンブル基準マージン
  content.py    選択OCRの行認識 + レイアウト検出 → 残差写真から字/図/写真を峻別
  pipeline.py   パラメータ確定 + 低確信ページにフラグ → SQLite 保存
  render.py     元画像から1パスでワープ → 検索可能PDF（G4/JBIG2 + 透明文字層）
  webui/        フラグ付きページのみレビュー → 上書き → 判断ログ → 再生成
```

## インストール

```bash
# 基本機能 + Yomitoku / NDL-OCR / Hybrid + WebレビューUI
pip install -e .[ocr,web]

# Apple Silicon: YomitokuのMPS高速化を含む構成
pip install -e ".[mac-ocr,web]"

# PaddleOCR-VL-1.6
pip install -e ".[paddle-vl,web]"

# Apple Silicon: PaddleOCR-VL認識をMLX/Metalで高速化
pip install -e ".[mlx-vl,web]"
```

`pyjbig2` は通常インストールで自動導入されます。追加のシステムコマンドは不要です。

## OCRエンジン

| 指定 | 検出器 | 認識器 | 主な用途 |
|---|---|---|---|
| `--ocr-engine hybrid` | Yomitoku系DBNet | NDL系認識器 | 既定。従来互換と図版レイアウト |
| `--ocr-engine yomitoku` | Yomitoku DBNet | Yomitoku PARSeq | Apple SiliconのMPS処理 |
| `--ocr-engine ndlocr` | NDL-OCR Lite DEIMv2 | NDL PARSeqカスケード | 高速なCPU処理 |
| `--ocr-engine paddle-vl-1.6` | PP-DocLayoutV3 | PaddleOCR-VL-1.6 | 文書構造を含むVLM解析 |

YomitokuとNDL-OCRは、それぞれ自身の検出器と認識器を使用します。`hybrid` は
Yomitoku系の文字検出とNDL系の文字認識を組み合わせ、Yomitokuの図版レイアウト検出も
実行します。

```bash
# 既定: Hybrid
hokusai-press run scan.pdf --out hybrid.pdf

# 各ネイティブOCR
hokusai-press run scan.pdf --out yomitoku.pdf --ocr-engine yomitoku --device mps
hokusai-press run scan.pdf --out ndlocr.pdf --ocr-engine ndlocr --device cpu

# PaddleOCR-VL-1.6をローカルCPUで実行
hokusai-press run scan.pdf --out paddle-vl.pdf \
  --ocr-engine paddle-vl-1.6 --runtime paddle --device cpu

# Apple Silicon: PP-DocLayoutV3はCPU、PaddleOCR-VL認識はMLX/Metal
python -m mlx_vlm.server \
  --model huggingfinger0/PaddleOCR-VL-1.6-8bit --port 8111
hokusai-press run scan.pdf --out book.pdf \
  --ocr-engine paddle-vl-1.6 \
  --runtime mlx --device cpu \
  --mlx-model huggingfinger0/PaddleOCR-VL-1.6-8bit

# 小容量の初代4bitモデル（PaddleOCR-VL v1。1.6ではありません）
python -m mlx_vlm.server \
  --model mlx-community/PaddleOCR-VL-4bit --port 8111
hokusai-press run scan.pdf --out book-v1.pdf \
  --ocr-engine paddle-vl-1.6 \
  --runtime mlx --device cpu \
  --mlx-model mlx-community/PaddleOCR-VL-4bit
```

レイアウトと文字認識を個別に指定する場合は、`--layout-engine` と
`--text-engine` を使用します。

```bash
hokusai-press run scan.pdf --out book.pdf --device npu      # バッチ解析+生成 (Intel NPU)
hokusai-press run scan.pdf --out book.pdf --layout-engine none --text-engine ppocr-v6 --runtime onnxruntime --device npu
```

## レビューと再生成

```bash
hokusai-press queue                                          # 要レビュー一覧
hokusai-press review                                         # レビューWeb UI
hokusai-press rebuild scan.pdf --doc scan.pdf --out book.pdf  # 修正後に再生成
hokusai-press learn --out model.json                        # 判断ログから学習
hokusai-press run scan.pdf --out book.pdf --learned-model model.json

# 保存済みOCR領域を再利用し、影除去・余白だけを再計算して再生成
hokusai-press remargin scan.pdf --out remargin.pdf --db hokusai.db

# 0始まりのページ指定でプレビュー処理、段階別時間も表示
hokusai-press run scan.pdf --out preview.pdf --pages 20-39 --profile
```

`--no-ocr` で OCR を省略し幾何処理＋ヒューリスティック分離のみでも動作します。
OCR/レイアウトは `--layout-engine` と `--text-engine` で別々に選択できます。
既定は既存互換の `hybrid` です。Paddle 系では
`pp-structurev3` / `paddle-vl` をレイアウトに、`ppocr-v6` を文字OCRに選択できます。
旧 `--ocr-engine` はまとめ指定の互換オプションとして残しています。
Intel NPU で実機確認済みの Paddle 経路は `--text-engine ppocr-v6 --runtime onnxruntime
--device npu` です。`pp-structurev3` と `paddle-vl` は環境ごとのbackend制約が
大きいため、追加依存を用途別に分けています。

レビュー UI はフラグ付きページのみを表示し、各ページの「解析オーバーレイ
（傾き補正後＋内容枠＋領域分類＋ノンブル）」と「出力プレビュー」を並べて表示。
ページ種別（bw/gray/color）の上書きに加え、写真領域・内容枠・ノンブル枠をドラッグで
追加・編集し、領域を削除できます。判断は特徴量つきでログされ、`rebuild` が修正済み
パラメータと入力原本からPDFを再生成します。

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
  のYomiToku DBNetモデルはCC BY-NC-SA 4.0です。利用・成果物の条件は上流モデルの
  ライセンスを確認してください。OCR/レイアウトは差し替え可能な構成です。
- [NDL-OCR Lite](https://github.com/ndl-lab/ndlocr-lite) は国立国会図書館が
  CC BY 4.0で公開しています。各OCRモデルを利用する場合は、モデルごとのライセンスも
  確認してください。
- JBIG2エンコーダ [pyjbig2](https://github.com/science-education/pyjbig2)
  はApache-2.0です。`jbig2enc 0.32`由来のlossless generic-region方式で、
  シンボル辞書を使わないため文字置換リスクのある非可逆JBIG2ではありません。

## 状態

実装済み・検証済み（217テスト、CI: Ubuntu/Windows/macOS・Python 3.12）:
- 非破壊パラメータモデル + JSON/SQLite 永続化
- deskew（確信度つき、縦横書き両対応）
- ノンブル基準マージン正規化（パリティ別・均一判型・クリップなし・ノンブル縦アンカー）
- 1パスレンダラ + 物理ページサイズ（実書籍で 4.70×7.27 inch 均一を確認）
- 多層 MRC（二値テキスト層 G4/JBIG2 + 写真層 JPEG、字は鮮明・写真は軽量）
- 字/図/写真の峻別: OCR行認識、Yomitoku/PP-DocLayoutV3/NDL-OCRの図版領域、
  残差インク写真ヒューリスティック
- Yomitoku、NDL-OCR Lite、Hybrid、PP-OCRv6、PaddleOCR-VL-1.6を切り替え可能
- Apple SiliconではYomitoku MPSとPaddleOCR-VL MLXサーバーに対応
- バッチ → 低確信ページのみレビュー（解析オーバーレイ+出力プレビュー、種別/領域編集）
  → 判断ログ → rebuild（入力原本から再加工の累積なしで再生成）
- 写真領域・内容枠・ノンブル枠のドラッグ編集、領域削除、OCRなしの余白再計算
- 判断ログからの page-kind 学習（最近傍重心、依存追加なし）→ 自動判定へ反映
- 検索可能テキスト層

次段: ノンブル基準マージンの綴じ方向対応（spine 整列）の本格 C++ 移植。
