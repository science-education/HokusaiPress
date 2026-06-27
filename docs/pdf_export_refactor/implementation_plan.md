# 実装計画: PDF/MRC エクスポートレイヤーのリファクタリング

## 1. 目的とスコープ
`hybrid_ocr.pdf_export` への実行時依存関係を排除し、`pikepdf` と `reportlab` を用いて PDF/MRC の生成を行う内部バックエンド（`internal`）を実装します。同時に、既存の `hybrid_ocr` バックエンドとの切り替えを可能にするファサードを構築し、動作の互換性と堅牢性を確保します。

## 2. 設計詳細

### 2.1. 内部バックエンドの画像エンコード (`hokusai_press/pdf_export/backend.py`)
- **CCITT G4 エンコーダ (`_encode_g4`)**:
  - `Pillow` で二値画像を TIFF (compression="group4") 形式でメモリ上に保存。
  - 保存された TIFF バイト列から、`StripOffsets` (273) と `StripByteCounts` (279) タグの情報を用いて生の CCITT G4 バイト列を抽出。
  - `pikepdf.Stream` を作成し、`/Filter /CCITTFaxDecode` および `/DecodeParms << /K -1 /Columns W /Rows H /BlackIs1 true >>` を指定。
- **JPEG エンコーダ (`_encode_jpeg`)**:
  - `cv2.imencode` を用いて、指定のクオリティで JPEG バイト列を生成。
  - `pikepdf.Stream` を作成し、`/Filter /DCTDecode` および `/ColorSpace /DeviceRGB` (カラー) または `/DeviceGray` (グレー) を指定。
- **暗黙のフォールバックの排除**:
  - サポート外のモードやパラメータが指定された場合、暗黙の代替エンコードを行わず、`UnsupportedCodecError` などの型付きエラーを直接発生させます。

### 2.2. フォント自動検出 (Font Discovery) とテキストレイヤー重ね合わせ
- **フォント自動検出ロジック**:
  - 明示的に `font_path` が指定された場合はそれを使用。
  - 未指定の場合、以下の候補パスを順に探索：
    1. `hybrid_ocr` パッケージがインストールされている場合、そのリソースディレクトリ (`hybrid_ocr/resource/MPLUS1p-Medium.ttf`)
    2. OSのシステムフォントディレクトリ (Windows: `%WINDIR%\Fonts\msgothic.ttc`, `msmincho.ttc` 等)
  - 適切な日本語フォントが見つからない場合は、明確な `MissingFontError` を発生させます。
- **縦書き・横書きのテキスト配置 (`build_text_overlay`)**:
  - reportlab を使用し、既存の配置ロジック（`direction=v` 時の1文字ずつの -90 度回転と y-up 座標系での配置）を完全に再現します。

### 2.3. ファサードとバックエンド選択 (`hokusai_press/pdf_export/__init__.py`)
- ファサードモジュールを提供し、以下の API を公開します。
  - `class SearchablePdfBuilder`:
    - 引数 `backend` (デフォルト `"auto"`) を追加。有効値は `"auto"`, `"internal"`, `"hybrid_ocr"`。
    - `auto` の場合、`hybrid_ocr.pdf_export` のインポート可否を明示的にチェックし、利用可能なら `hybrid_ocr` を、不可能なら `internal` を使用。
    - 明示指定されたバックエンドが利用できない場合は `BackendUnavailableError` を発生。
  - `encode_page_pdf`, `build_text_overlay`, `decide_page_mode`
- `mrc.py` における PDF エンコードのインポート元を、この新モジュールに切り替えます。
- `mrc.py` 内に定義されている `_encode_gray_flate_page_pdf` を `hokusai_press/pdf_export/backend.py` に移動（または共有）し、再利用します。

## 3. テスト計画
`tests/test_pdf_export_refactor.py` に以下の検証を追加します：
- `pikepdf` による再オープン可能性とメタデータの整合性
- 生成された PDF ページの正確な `MediaBox` (1px = 1pt 物理スケール)
- `CCITTFaxDecode` / `DCTDecode` / `FlateDecode` フィルタの存在およびパラメータ検証
- 各モード (`bw`, `gray`, `color`) のエンコード動作確認
- 無効な入力 (空の画像、不整合な画像など) に対する型付きエラーの送出検証
- 空のテキストページが含まれる場合の重ね合わせ動作
- 日本語の横書き・縦書きテキストのレイヤー重ね合わせの正確性
- `hybrid_ocr.pdf_export` が利用できないモック環境での `internal` バックエンドの正常動作確認
