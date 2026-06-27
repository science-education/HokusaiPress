# タスクリスト: PDF/MRC エクスポートレイヤーのリファクタリング

- [x] 1. リポジトリおよび `hybrid_ocr.pdf_export` の依存関係の調査
- [x] 2. タスクリストおよび実装計画ドキュメントの作成
- [ ] 3. 内部バックエンド (`hokusai_press.pdf_export` サブモジュール) の実装
  - [ ] 3.1. G4 画像ストリームエンコーダ (`_encode_g4`) の実装
  - [ ] 3.2. JPEG 画像ストリームエンコーダ (`_encode_jpeg`) の実装
  - [ ] 3.3. フォント自動検出 (Discovery) ロジックおよび `MissingFontError` の実装
  - [ ] 3.4. 横書き・日本語縦書きテキストレイヤー重ね合わせ (`build_text_overlay`) の実装
  - [ ] 3.5. ファサードクラス `SearchablePdfBuilder` とバックエンド選択 (`auto` / `internal` / `hybrid_ocr`) の実装
- [ ] 4. `mrc.py` のインポート元のリファクタリング (内部バックエンドへの移行)
- [ ] 5. 包括的なテスト (`tests/test_pdf_export_refactor.py`) の実装
- [ ] 6. 全 pytest テストの実行と `compileall` による検証
- [ ] 7. ローカルコミットの作成
