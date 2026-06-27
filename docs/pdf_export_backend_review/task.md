# タスク概要: PDFエクスポートバックエンドのセカンドパスコードレビューと修正

## 目的
`refactor/pdf-export-backend` ブランチにおけるコミット内容を詳細に検証し、指摘された懸念点およびバグを修正した上でローカルコミットを作成する。

## 要件・チェックポイント
1. **`backend='auto'` の挙動修正**
   - 常に `internal` を優先的に選択することでランタイム依存（`hybrid_ocr`）を排除する。
   - `compress='jbig2'` が要求され、かつ `pyjbig2` パッケージが利用できない場合のみ `hybrid_ocr` にルーティングする。
2. **G4 DecodeParms `BlackIs1` の極性検証**
   - 非対称な白黒テスト画像を用いて、Pillow TIFF raw strip と PDF レンダラー側での極性が一致しているかを検証する。
   - 検証の結果、現行の `BlackIs1=True` が極性一致として正しいことを確認する。
3. **JPEG RGB の色保存検証**
   - `cv2.imencode` を用いた JPEG ストリームと PDF `/DeviceRGB` への出力において、BGRからRGBへの変換時に色相が正しく維持されていることを検証する。
4. **ReportLab フォント探索のポータビリティ向上**
   - Windows 以外の環境（Ubuntu CI 等）で `hybrid_ocr` や Windows フォントがない場合にも、Linux/macOS のシステムフォントパスからフォントを発見できるようにする。
   - 法的ライセンスの観点からも安全に CI を通過できる状態を確保する。
5. **テストにおける `hybrid_ocr` 依存の排除と MRC エンドツーエンド検証**
   - `tests/test_mrc.py` の先頭にある `pytest.importorskip("hybrid_ocr")` を削除し、`hybrid_ocr` がない環境でも MRC のテストが走るようにする。
   - `MrcPageBuilder` のテストで `hybrid_ocr` のインポートを `hokusai_press.pdf_export` に置き換える。
   - `tests/test_pdf_export_refactor.py` に `hybrid_ocr` 不在をモックした上での MRC 結合テストを追加する。
6. **不要な API 面の削減と検証**
   - `SearchablePdfBuilder` および `decide_page_mode` などが `hybrid_ocr` 側の API 構成と一致していることを調査・担保し、不要な API 定義を排除する。
7. **入力バリデーションの強化**
   - 入力画像 (`img_bgr`) が `np.ndarray` であること、かつ空配列や不正な次元数（1次元など）でないことをチェックするバリデーションを追加する。
8. **その他品質管理**
   - インポートスタイルの確認および未使用インポートのクリーンアップ。
   - 日本語の縦書き・横書き処理の挙動が維持されていることの確認。
   - `compileall`、`pytest`、`git diff --check` がすべて正常に通過すること。
   - 修正内容をローカルコミットとしてコミットし、リモートへの Push は行わない。
