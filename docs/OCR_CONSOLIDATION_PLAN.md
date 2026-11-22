# OCR関連ツールの整理と統合アプリの計画

作成 2026-11-22。対象は、これまで作ってきたスキャン書籍の補正・OCRのツール一式。

## 1. 棚卸し（いま手元にあるもの）

| ツール | 場所 | 役割 | 状態 | ライセンス |
|---|---|---|---|---|
| ScanTailorAdvancedTATEGAKI | `dev\ScanTailorAdvancedTATEGAKI`（＋`-trackB2`） | GUIで1冊ずつ補正・二値化（600dpi・mixed）。DEIM／YomiToku検出をコンテンツ選択に組み込み済み | 前回63冊の「本文二値化」はこれで作成。手作業が必要。作業ツリーに未コミットの変更あり | GPLv3 |
| HokusaiPress | `dev\HokusaiPress`（main = `acc88a4`） | 自動の補正（傾き・余白・影・判型統一）、二値化／MRC、OCR、レビューUI、判断の学習、ノンブル | 機能は一番多い。OCRエンジンを差し替え可能（hybrid／yomitoku／**ndlocr**／paddle-vl、Macは MPS／MLX） | GPLv3 |
| HokusaiPress-ocr-paddle | 同リポジトリの worktree | Paddle系アダプタの開発場所 | main にマージ済みで、もう不要 | — |
| HokusaiPress_review | `dev\HokusaiPress_review` | 同じリポジトリの別クローン | main と同じ内容。重複 | — |
| Yomitoku_NDL-OCR-Lite（hybrid） | `dev\Yomitoku_NDL-OCR-Lite` | YomiToku DBNet検出＋NDL PARSeq認識。検索可能PDF（G4／JBIG2、フォントのフォールバック、ページ寸法の修正済み） | 安定。DBNetが固定形状でNPUに載る（CPU比2.4倍） | CC BY-NC-SA（YomiToku由来） |
| NDL-OCR-Lite-NPU | `dev\NDL-OCR-Lite-NPU` | NDL純正のDEIM＋PARSeq 3段。PARSeqのみNPU可（書籍規模で約1.45倍） | 2026-11-22に PDF入力と `--pdf`（元PDFに文字層を重ねる）を追加（ブランチ `feat/searchable-pdf`、未push） | CC BY 4.0 |
| ndlocr-lite-web-ai | `dev\ndlocr-lite-web-ai` | ブラウザだけで動くNDL OCR＋AI校正UI | 3月以降は止まっている | — |
| AI校正（3者判定） | 作業場 `dev\tmp\ocr0915\gcheck` は**消えている** | Geminiが怪しい行を指摘 → Codexが画像だけを見て読み直す → 一致したら採用 | 手順はメモリにだけ残る。コードは失われた | — |
| YomiToku一括スクリプト | `補正済み\00_倉庫\yomitoku_pdf_batch.sh` | 前回63冊のOCR | 二値画像をカラーJPEGに作り直してしまう（劣化、例8.6MB→45MB） | — |

## 2. 分かっている事実（実測）

**認識精度**（`C:\tmp\acc-bench`、正解付き14頁で2026-11-22に再計算。空白を除いた文字誤り率）

| エンジン | shinto（素直な縦書き） | chufu（難しい頁） |
|---|---|---|
| NDL（DEIM＋PARSeq） | 1.6% | 9.4% |
| hybrid CPU／NPU | 1.9%／2.0% | 9.7%／9.6% |
| YomiToku（読み順補正あり） | 1.6% | 9.8% |

→ **認識精度は3者でほぼ同じ**。差が出るのは、行の検出漏れ（hybridは行の脱落・縦中横の取りこぼしが典型）、読み順、速度、出力PDFの質。

**速度**（この機 Core Ultra 7 155U）: NDL CPU 約1.9〜3.7秒/頁、hybrid NPU 約2.3〜3秒/頁、hybrid CPU 約5.6〜8.4秒/頁、YomiToku フル 約20秒/頁、PaddleOCR-VL 約70秒/頁。NDLのDEIMはNPUに載らない。

**NDLの強み**: DEIMが17クラスのレイアウト（図・表・ノンブル・柱など）を出す。CC BY 4.0 で、成果物に非商用の制約がつかない。

**HokusaiPressの現状の不具合**（2026-11-22に確認）: `--no-ocr` だと内容枠の推定を誤り、表紙のタイトル文字を余白として白く塗ってしまう（`img20261121_0002` p1）。OCRを有効にすると正しく出る。

## 3. 問題点（整理が必要な理由）

1. **同じ機能が3〜4か所に重複**: 検索可能PDFの文字層（hybrid `pdf_export.py`、HokusaiPress `pdf_export/`、NDL `searchable_pdf.py`）、二値化（ScanTailor、HokusaiPress `mrc.py`、hybrid `binarize`）、PDFの読み込み。
2. **入口が多く、手順が人の記憶頼み**: 元データ → 本文二値化 → OCR済、の流れを毎回思い出す必要がある（今回それが起きた）。
3. **Python環境が分かれている**: メインの3.14ではNPU／YomiTokuが動かない。実運用は `NDL-OCR-Lite-NPU\.venv-openvino`（3.12）を間借りしている。
4. **評価データと校正コードが揮発する場所にある**: `C:\tmp` や `dev\tmp` に置いたベンチ・校正スクリプトが消えている。
5. **ローカルのcheckoutが古いことに気づけない**: 今回、HokusaiPressのローカルが origin/main より9コミット遅れたまま使われていた。
6. **ライセンスが混在**: hybridの検出器（YomiToku）は非商用（CC BY-NC-SA）。既定エンジンがhybridのままだと、成果物すべてが非商用扱いになる。

## 4. 目標: HokusaiPress を唯一のアプリにする

新しいアプリは作らない。いちばん機能が多く、エンジンを差し替える仕組みもあるHokusaiPressに集約する。他のリポジトリは「部品の供給元」か「保存のみ」にする。

### 4.1 処理の流れ（1コマンドで完結）

```
元データPDF ──▶ 補正（傾き・余白・影・判型）──▶ 二値化／MRC ──▶ OCR ──▶ (任意) AI校正 ──▶ 出力
                                                                     ├ 本文二値化PDF（文字層なし）
                                                                     ├ OCR済PDF（同じ画像＋文字層）
                                                                     └ テキスト・JSON・判断ログ
```

- **既定のOCRエンジンをNDL（`--ocr-engine ndlocr`）に変える**。精度は同等で、CPUで最速。レイアウト分類も使え、ライセンスもきれい。hybrid（NPU）は「速度優先」の選択肢として残し、PaddleOCR-VLは難しい頁だけの選択肢にする。
- **段階ごとの成果物を一度に出す**（`--emit binarized,ocr`）。今の「本文二値化」「OCR済」のフォルダ運用をそのまま再現する。
- **OCRだけをやり直すモード**（`hokusai-press ocr <pdf>`）。補正済みのPDFの画像をそのまま残し、文字層だけを付け替える（今回NDLに追加した重ね方式と同じ）。既存の63冊を、劣化させずにNDLでOCRし直すこともできる。
- **ファイル名に処理の印を付ける**（例 `_HP`）。あわせて、奥付・標題紙のOCRからNDL書誌を引いて書名に改名する機能を付ける（Hondanaで実績のある方法）。

### 4.2 部品の一本化

| 機能 | 残すもの | 片付けるもの |
|---|---|---|
| 文字層の書き込み | 1つのモジュールにまとめる（NDL `searchable_pdf.py` の重ね方式に、hybridのフォントフォールバック・互換漢字の畳み込みを統合） | hybrid・HokusaiPress の重複コード |
| 画像の符号化 | HokusaiPress の MRC（G4／JBIG2＋JPEG） | hybrid の `encode_page_pdf` |
| 二値化 | HokusaiPress `binarize_bw`（Otsu＋裏移りの2信号判定） | hybrid の `binarize`（OCR入力用にだけ残す） |
| NDLエンジン | NDL-OCR-Lite-NPU をパッケージとして依存利用 | — |
| AI校正 | 3者判定の手順を `hokusai_press/proof/` に作り直す（任意の工程） | 消えた gcheck |

### 4.3 環境・運用

- 実行用venvを1つにする: HokusaiPress専用の Python 3.12 venv（OpenVINO／NDL／hybridを同居）。`NDL-OCR-Lite-NPU` の venv の間借りをやめる。
- 起動時に、ローカルが origin より古ければ警告する（`hokusai-press doctor`）。
- 評価セット（正解付き頁）をリポジトリ内の `bench/` に移し、エンジンごとの誤り率・検出漏れ率を1コマンドで出す。CIでは数頁だけ回帰テストする。
- 長時間バッチは切り離したプロセスとログで回す（今回の `run_hp.ps1` の形）。中断した冊から再開できるようにする（出力済みはスキップ）。

### 4.4 保存のみにするもの

- **ScanTailorAdvancedTATEGAKI**: HokusaiPressの自動補正で品質が足りない本の「手動の逃げ道」として残す。新機能は入れない。
- **Yomitoku_NDL-OCR-Lite**: hybridエンジンの供給元として維持。PDF出力の開発はやめる。
- **ndlocr-lite-web-ai**: 単発の画像をブラウザで読む用途として残す（書籍の一括処理はしない）。
- `HokusaiPress-ocr-paddle` の worktree と `HokusaiPress_review` クローンは削除してよい（確認のうえ）。

## 5. 段階計画

| 段階 | 内容 | 完了の条件 |
|---|---|---|
| 0 準備 | 専用venv、`doctor`、bench をリポジトリへ移す、NDL `feat/searchable-pdf` をpush・マージ | `bench` でNDL／hybridの誤り率が上の表と同じ値で再現する |
| 1 NDLを既定に | `--ocr-engine ndlocr` をこの機で実測（速度・頁ごとの検出漏れ）。問題なければ既定にする | 新規18冊のうち3冊×30頁で目視確認し、表紙・目次・図版頁で文字が消えない |
| 2 `--no-ocr` の不具合 | 内容枠の推定失敗を「全面−細い縁」へ倒す（残課題C） | `img20261121_0002` p1 の題字が残る。既存テストが緑 |
| 3 段階別出力とOCRだけのやり直し | `--emit`、`ocr` サブコマンド、文字層モジュールの統合 | 二値化PDFとOCR済PDFの画像がバイト単位で一致。古い文字層が残らない |
| 4 命名 | 印の付与、書誌からの書名への改名（候補を出し、人が確定） | 18冊で書名候補の正解率を記録 |
| 5 AI校正 | 3者判定を任意の工程として作り直す | 1冊で、採用・保留の件数と無作為抽出の目視の正解率を記録 |
| 6 片付け | 重複コードの削除、保存のみのリポジトリのREADMEに「保存のみ」と明記 | 各リポジトリのREADMEから、どこを使えばよいかが一目で分かる |

各段階は、成果物の中身（画素・抽出したテキスト）で確かめる。テストが緑、という自己申告だけで完了にしない。

## 6. 決めていただきたいこと

1. 統合先を **HokusaiPress** にしてよいか（新しいアプリは作らない）。
2. 既定のOCRエンジンを **NDL** に切り替えてよいか（hybridは速度優先の選択肢として残す）。
3. ScanTailorAdvancedTATEGAKI を「保存のみ」にしてよいか。
4. Mac（Apple Silicon）対応をどこまで維持するか（現在は YomiToku の MPS と PaddleOCR-VL の MLX）。
