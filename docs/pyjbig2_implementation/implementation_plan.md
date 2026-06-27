# pyjbig2 実装計画 (M2 / M3 マイルストーン)

M1（ピュアPython版）での機能・正確性検証が完了したため、次のステップとして**M3（Cythonによる高速化）**および**M2（実データによるコーパス評価）**に進みます。

ピュアPython実装のままでは M2 の「970頁の処理」に長時間を要するため、先に M3 の Cython 化を実施し、その高速化されたコアを用いて M2 のコーパス評価を行う方針とします。

## User Review Required

> [!IMPORTANT]  
> 1. **Cython化の実装方針**: ピュアPython実装である `_reference.py` を元に、C言語レベルの型アノテーションを付与した `_cython_mqcoder.pyx` を作成し、C拡張としてビルドします。`api.py` では C拡張が利用可能な場合はそちらを優先し、利用できない場合は `_reference.py` にフォールバックする設計とします。
> 2. **コーパス評価**: `C:\tmp\tmp0613` に含まれる PDF から画像を抽出し、可逆エンコードを実行します。目標である「300dpi A4 二値 1頁を 1コア ≤ 300ms」を達成できているか、また圧縮サイズが既存ツール (`jbig2enc` など) と比較して妥当かを検証します。

## Proposed Changes

### 1. M3: Cython 高速化コアの実装

#### [MODIFY] `C:\Users\user\dev\pyjbig2\pyproject.toml`
- ビルド依存関係に `Cython` および `setuptools` を追加します。

#### [NEW] `C:\Users\user\dev\pyjbig2\setup.py`
- Cython 拡張モジュール (`_cython_mqcoder.pyx`) をビルドするためのスクリプトを追加します。

#### [NEW] `C:\Users\user\dev\pyjbig2\pyjbig2\_cython_mqcoder.pyx`
- `_reference.py` を移植し、以下の処理を静的型付け (C型の利用) で高速化します：
  - `MQEncoder` の `ENCODE`, `RENORME`, `BYTEOUT`
  - `GenericRegionEncoder` における画素ループと 16ビットコンテキスト生成
- C言語の配列・ポインタ操作に落とし込むことでループのオーバーヘッドを極小化します。

#### [MODIFY] `C:\Users\user\dev\pyjbig2\pyjbig2\api.py`
- エンコード処理時、可能であれば `_cython_mqcoder` を使用し、インポートに失敗した場合は `_reference.py` を使用するようにフォールバック処理を追加します。

### 2. M2: コーパス評価スクリプトの実装

#### [NEW] `C:\Users\user\dev\pyjbig2\tests\benchmark_corpus.py`
- `C:\tmp\tmp0613` にある PDF から数ページ（または全ページ）を抽出し、以下を計測・検証するスクリプト：
  - 処理時間 (1ページあたり300ms以内を達成しているか)
  - Cython版とPython版の出力が bit-exact で一致するか（一部ページでテスト）
  - 出力された JBIG2 データのサイズ

## Verification Plan

### Automated Tests
- `pytest` を再実行し、Cython 拡張を用いたエンコードが既存のテストを通過することを確認します。

### Manual Verification
- `benchmark_corpus.py` を実行し、コンソールに出力される処理時間および圧縮結果のレポートを確認します。
- 必要に応じて、実行結果のレポートを Artifact (または `walkthrough.md`) に記録し報告します。
