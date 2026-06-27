# 修正内容の確認 (Walkthrough): PDF/MRC エクスポートレイヤーのリファクタリング

## 1. 変更概要
本リファクタリングでは、`hybrid_ocr.pdf_export` への実行時依存関係を排除し、`pikepdf` と `reportlab` を使用して PDF/MRC の生成を直接行う内部バックエンド (`internal`) を構築しました。これにより、OCR エンジンがインポートできない環境でも、PDF/MRC などのエクスポート機能が独立して動作するようになりました。

## 2. 主な修正内容

### 2.1. 新規モジュール `hokusai_press.pdf_export` の作成
- `exceptions.py`:
  - 型付きエラー `UnsupportedCodecError`, `MissingFontError`, `BackendUnavailableError` を定義。
- `backend.py`:
  - `encode_page_pdf`:
    - `bw` モードにおいて、`Pillow` で画像を TIFF G4 にエンコードし、そこから生の G4 バイト列を抽出して `pikepdf.Stream` (`/CCITTFaxDecode` フィルタ) に格納するロジックを実装。
    - `gray`/`color` モードにおいて、OpenCV で JPEG エンコードし、`pikepdf.Stream` (`/DCTDecode` フィルタ) に格納するロジックを実装。
  - `build_text_overlay`:
    - 日本語フォントの自動検出 (Discovery) ロジックを実装（OS フォントディレクトリや `hybrid_ocr` のリソースから探索）。
    - reportlab を使用して、縦書き (`direction=v` 時の一文字ごとの -90 度回転配置) および横書きテキストの重ね合わせを再現。
  - `_encode_gray_flate_page_pdf`:
    - `mrc.py` から既存の Flate エンコーダを移動し、再利用可能に。
- `__init__.py`:
  - ファサード `SearchablePdfBuilder` を提供。
  - `backend` 引数 (`"auto"`, `"internal"`, `"hybrid_ocr"`) によるバックエンド選択機能を実装。デフォルトは `auto` で、`hybrid_ocr` が利用可能な場合はそちらを優先、利用不可能な場合は `internal` を透過的に使用。
  - 明示的に `hybrid_ocr` が指定されたが利用できない場合は `BackendUnavailableError` を送出し、サポート外のコーデックが指定された場合は `UnsupportedCodecError` を送出するよう例外設計を徹底。

### 2.2. `mrc.py` のインポートおよびエンコード処理の移行
- `hybrid_ocr.pdf_export` からのインポートを `hokusai_press.pdf_export` へ移行。
- `mrc.py` 内に定義されていた `_encode_gray_flate_page_pdf` を削除し、`hokusai_press.pdf_export` からインポートして再利用する形に集約。

### 2.3. `pyproject.toml` 依存関係の更新
- `hybrid-ocr` がインストールされていなくても内部バックエンドが自律的に動作するよう、`dependencies` に `"Pillow"` および `"reportlab"` を明示的に追加。

### 2.4. 包括的なテストの追加
- `tests/test_pdf_export_refactor.py` を追加し、以下の項目を網羅：
  - `pikepdf` による PDF の再オープンとメタデータの整合性
  - `MediaBox` が物理スケール (1px = 1pt) と完全に一致していることの検証
  - 各モード (`bw`, `gray`, `color`) での正しい PDF フィルタ (`/CCITTFaxDecode`, `/DCTDecode`) およびデコードパラメータの検証
  - 空のテキストページの挙動
  - 縦書き（文字回転）・横書きテキスト配置の正確性
  - フォント自動検出エラー (`MissingFontError`)
  - `hybrid_ocr` インポート不可環境をモックした上での `internal` バックエンドの正常動作確認

## 3. テスト実行結果
- プロジェクト全体の `pytest` テスト (計 196 個) はすべて成功 (`196 passed`)。
- `python -m compileall src` を実行し、構文・コンパイル上の問題がないことを確認。
