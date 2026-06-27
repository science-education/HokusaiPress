# pyjbig2 実装タスク (M2 / M3 マイルストーン)

- `[x]` M3: Cython 高速化コアの実装
  - `[x]` `pyproject.toml` へ Cython/setuptools の追加
  - `[x]` `setup.py` の作成
  - `[x]` `_cython_mqcoder.pyx` の実装
  - `[x]` `api.py` の更新 (Cythonモジュールのロード)
  - `[x]` `pytest` による bit-exact の検証
- `[x]` M2: コーパス評価
  - `[x]` `benchmark_corpus.py` の作成
  - `[x]` `C:\tmp\tmp0613` の PDF 群からの画像抽出処理
  - `[x]` 可逆性、処理時間 (≤300ms/頁)、ファイルサイズの計測
  - `[x]` レポート (`walkthrough.md`) の作成
