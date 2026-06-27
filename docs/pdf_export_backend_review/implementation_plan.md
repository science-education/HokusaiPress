# 実実装計画: PDFエクスポートバックエンドの修正と検証

## 1. 修正方針

### 1.1 `backend='auto'` の優先順位の見直し
- **目的**: `hybrid_ocr` へのランタイム依存を取り除き、能力ベースでのルーティングを行う。
- **実装詳細**:
  - `decide_page_mode` および `build_text_overlay` において、`backend='auto'` の場合は常に `internal` バックエンドを使用する。これらの処理は OpenCV/ReportLab で完結するため、`hybrid_ocr` にフォールバックする必要がない。
  - `encode_page_pdf` および `SearchablePdfBuilder` において、`compress='jbig2'` が指定され、かつ `pyjbig2` パッケージがインポートできない場合に限り、`hybrid_ocr` バックエンドへフォールバックして処理する。それ以外の場合は `internal` を使用する。

### 1.2 ReportLab フォント探索のポータビリティ向上
- **目的**: Ubuntu CI などの非Windows環境で Windows フォントや `hybrid_ocr` がない場合でも、日本語フォントを自動検出できるようにする。
- **実装詳細**:
  - Windows のフォントパスに加え、Linux (Ubuntu等) の標準フォントパス（`/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf` 等、Noto CJK、Takao、VL Gothic、Droid 等）および macOS の標準フォントパスを候補リストに追加し、存在するものを自動的に選択する。

### 1.3 `tests/test_mrc.py` の `hybrid_ocr` 依存の排除
- **目的**: `hybrid_ocr` がない環境でも MRC 関連のすべてのユニットテストを走らせる。
- **実装詳細**:
  - `pytest.importorskip("hybrid_ocr")` を削除。
  - `test_tint_posterized_flate_beats_jpeg_on_flat_halftone` テスト内で `hybrid_ocr.pdf_export` から `encode_page_pdf` をインポートしていた箇所を、自前の `hokusai_press.pdf_export` からのインポートに修正。

### 1.4 入力バリデーションの強化
- **目的**: 不正な画像データが渡された場合に早期にエラーを発生させる。
- **実装詳細**:
  - `img_bgr` に対し、`isinstance(..., np.ndarray)` による型確認、サイズ確認、および次元数の確認（2次元以上であること）を行う。
  - `_encode_g4` および `_encode_jpeg` においても、引数配列の形状とデータ型（`uint8`）のバリデーションを挿入する。

### 1.5 単体テストの追加
- **目的**: 修正した挙動の正しさを保証する。
- **実装詳細**:
  - `tests/test_pdf_export_refactor.py` に `test_mrc_page_builder_internal` を追加し、`hybrid_ocr` をモックで無効化した状態での MRC エンドツーエンド出力を検証する。
  - `test_backend_auto_routing` を追加し、`backend='auto'` の優先選択および capability ルーティングが期待通り動作することを検証する。
  - `test_input_validation` を追加し、画像データに対するバリデーションが正しく機能することを検証する。

## 2. 実行手順

1. `src/hokusai_press/pdf_export/__init__.py` の修正
2. `src/hokusai_press/pdf_export/backend.py` の修正
3. `tests/test_mrc.py` の修正
4. `tests/test_pdf_export_refactor.py` の修正
5. `pytest` による全単体テストの実行・確認
6. `compileall` によるコンパイルチェック
7. `git diff --check` によるフォーマット確認
8. ローカルコミットの作成
