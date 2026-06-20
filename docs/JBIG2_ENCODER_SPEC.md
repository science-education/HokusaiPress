# pyjbig2 — 自己完結 JBIG2 (generic, lossless) エンコーダ 仕様書

別AI／別エンジニアへ委譲可能な実装仕様。HokusaiPress の WSL 依存(jbig2enc)を
置き換える「pip で入る JBIG2 エンコーダ」を新規ライブラリとして作る。

## 0. ゴールと非ゴール

ゴール:
- 1bpp 二値画像を **JBIG2 generic region（可逆, lossless）** で符号化し、
  **PDF の `/JBIG2Decode` ストリーム**として埋め込めるバイト列を返す。
- **外部実行ファイル・Leptonica 不要**。`pip install pyjbig2` だけで動く。
- 既存 `hybrid_ocr.pdf_export._page_pdf_jbig2` の差し替え先になる API。

非ゴール（実装しない。やらないことを明示）:
- symbol/text region（文字辞書）— OCR文字化けリスク回避方針と一致。
- halftone region, refinement region, MMR(=G4) coding。
- 算術積分符号(IAID等)。generic の MQ coder 以外の算術手続きは不要。

## 1. 準拠規格と参照

- **ITU-T T.88 (2018)** "Information technology – Lossy/lossless coding of
  bi-level images (JBIG2)"。無償公開。実装根拠はこれ一本。
  - Annex E: 算術符号化（MQ coder）。Qe 確率表・INITENC/ENCODE/BYTEOUT/FLUSH。
  - §6.2: generic region decoding procedure（テンプレート, AT画素, TPGDON）。
  - §7.2: segment header 構造。
  - §7.4.1: region segment information field。
  - §7.4.6: generic region segment。
  - Annex D: file/embedded organization（PDF は embedded = ファイルヘッダ無し）。
- PDF 連携: PDF 32000-1 §7.4.7 "JBIG2Decode filter"。
- **clean-room**: jbig2enc のソースは見ない（参照は T.88 本文のみ）。
  ライセンス的には Apache-2.0 で問題ないが、由来を T.88 に限定して
  本ライブラリを **Apache-2.0 または MIT** で出す（GPLv3 互換）。

## 2. アーキテクチャ

```
pyjbig2/
  _mqcoder.{pyx|rs}   # MQ 算術符号化器（ホットパス, compiled）
  _generic.{pyx|rs}   # generic region のコンテキスト生成＋画素ループ（compiled）
  segments.py         # segment header / region info の組立（純Python, 低頻度）
  api.py              # encode_generic(), page_pdf_jbig2()
  _reference.py       # 純Python版 MQ+generic（検証用オラクル, 遅くてよい）
tests/
```

方針: **まず `_reference.py`（純Python）で bit-exact を作り**、独立デコーダで
round-trip 検証 → その後 `_mqcoder`/`_generic` を Cython か Rust(pyo3) に移植し、
両者の出力バイト列が一致することを回帰テストで固定する。

## 3. 実装すべきアルゴリズム

### 3.1 MQ arithmetic encoder (T.88 Annex E)
- 状態: A(16bit), C(32bit), CT, BP, バッファ。
- Qe 表 47 エントリ `(Qe, NMPS, NLPS, SWITCH)`（T.88 Table E.1 をそのまま定数化）。
- コンテキスト: 各 cx は `(I: 0..46, MPS: 0/1)` を保持する配列。generic は
  GBTEMPLATE に応じた個数（template0 = 2^16 contexts）。
- 手続き: `INITENC`, `ENCODE(D, cx)`（→ codeMPS/codeLPS＋RENORME）,
  `BYTEOUT`（0xFF スタッフィングとキャリー伝播）, `FLUSH`(SETBITS)。
- 検証: T.88 が用意する **算術符号化テスト列**で出力が規格一致すること。

### 3.2 Generic region (T.88 §6.2)
- 入力: `bitmap` (H×W, 値 {0,1}、**JBIG2 は 1=黒**)。
- GBTEMPLATE: まず **template 0**（10画素コンテキスト）固定で可。
- AT 画素: 公称位置（nominal）に固定（A1=(+3,-1), A2=(-3,-1), A3=(+2,-2),
  A4=(-2,-2) の規格公称値）。適応探索は将来拡張。
- **TPGDON = 0**（typical prediction 無効）で開始。可逆性に影響しないため、
  サイズ最適化として後段で TPGDON=1 を追加。
- ラスタ順に各画素の CONTEXT を既符号化近傍から組み、`ENCODE(pixel, CONTEXT)`。
- 境界外画素は 0 とみなす（規格通り）。

### 3.3 Segment 組立（embedded / PDF 用）
PDF `/JBIG2Decode` は **embedded organization**（ファイルヘッダ・EOFセグメント無し）。
1ページ1 generic region の最小構成として、以下を順に連結:

1. （任意だが互換性のため推奨）**page information segment**（type 48,
   §7.4.8）: ページ幅・高さ・解像度・flags。
2. **immediate lossless generic region segment**（**type 39**, §7.4.6）:
   - segment header（§7.2.1）:
     - segment number(4B)
     - flags(1B): 下位6bit = type(39), bit6 = page assoc size, bit7 = deferred
     - referred-to count & retention flags(1B, 0 参照)
     - segment page association(1B=1)
     - segment data length(4B) = 続くデータ長
   - segment data:
     - region segment information field(17B, §7.4.1): width,height,X=0,Y=0,
       flags(1B: external combination operator=OR/REPLACE)
     - generic region flags(1B, §7.4.6.2): MMR=0, GBTEMPLATE(2bit), TPGDON
     - AT pixels（template0 なら 4 対 = 8B, signed int8）
     - 算術符号化データ（3.2 の出力）

> 実装上の最重要検証点: **segment header の可変長フィールド**（referred-to の
> サイズは segment number に依存）と **page association サイズ**。ここを
> 独立デコーダの round-trip で必ず固める。

### 3.4 PDF 埋め込み
既存 `_page_pdf_jbig2`（pikepdf）を踏襲:
```python
image.write(data, filter=Name("/JBIG2Decode"))
image.BitsPerComponent = 1
image.ColorSpace = Name("/DeviceGray")
image.Width, image.Height = w, h
```
ビット極性: JBIG2 は 1=黒、PDF DeviceGray は 0=黒。既存 jbig2enc 出力が現行
コードで正しく表示できている事実に合わせる（必要なら `/Decode [1 0]` で反転）。
**現行 `_page_pdf_jbig2` の表示が正しい設定を真とし、それに一致させる**。

## 4. 公開 API

```python
def encode_generic(
    bitmap: "np.ndarray",      # (H,W) bool または uint8{0,1}; True/1 = 黒
    *, template: int = 0, tpgdon: bool = False,
) -> bytes:
    """1 つの generic region を embedded JBIG2 セグメント列として返す。"""

def page_pdf_jbig2(bitmap: "np.ndarray") -> bytes:
    """単一ページ PDF（JBIG2Decode 画像）を返す。_page_pdf_jbig2 互換。"""
```
追加要件: 入力 dtype 検証、0/255 入力も受理（>0 を黒とみなす）、空/全黒/全白の
退化ケース。

## 5. 検証・適合性（受け入れ条件）

1. **可逆性（最重要）**: encode → 独立デコーダ → 入力と **bit-exact 一致**。
   - 独立デコーダ: `jbig2dec`、または PDF にして **pdf.js / Ghostscript / pikepdf+pdfium**
     でラスタライズし元 bitmap と比較。少なくとも 2 実装で一致を要求。
2. **MQ 規格適合**: T.88 算術符号化テスト列で出力一致（単体）。
3. **実コーパス**: HokusaiPress 実データ（`C:\tmp\tmp0613` 全5冊・970頁）の
   二値ベース層を符号化し、(a) 全頁可逆、(b) サイズを G4 / jbig2enc と比較。
   - 目標: **サイズは jbig2enc `-p` の ±5% 以内**、G4 より小さい。
4. **退化ケース**: 1×1, 全白, 全黒, 1px線, 縦長極端アスペクト。
5. **性能**: 300dpi A4 二値 1 頁を **1コア ≤ 300ms**（compiled コア）。
   純Python reference は機能検証専用（遅くてよい）。

## 6. マイルストーン（委譲単位）

- **M1 正しさ**: 純Python MQ + generic(template0,TPGDON=0) + segment + PDF。
  synthetic と実頁数枚で round-trip bit-exact（jbig2dec & pdfium）。
- **M2 コーパス**: 970 頁で可逆＋サイズレポート（vs g4, vs jbig2enc）。
- **M3 性能**: コアを Cython/Rust に移植、wheel 化（cibuildwheel: win/mac/linux,
  cp310-cp313）。reference と出力一致を回帰固定。性能目標達成。
- **M4 仕上げ**: TPGDON=1・template 選択でサイズ最適化、PyPI 公開、
  HokusaiPress 統合（`_page_pdf_jbig2` を pyjbig2 へ。**jbig2 バイナリ未導入時の
  G4 フォールバックは維持**）。

## 7. リスクと対策

- **segment header のビット詰め誤り** → M1 から独立デコーダ round-trip で常時検証。
- **MQ の FLUSH/終端・0xFF スタッフィング** → T.88 テスト列で単体固定。
- **ビット順**: JBIG2 データは MSB-first パッキング。バイト境界の扱いを明示テスト。
- **純Python 性能** → 最初から hotspot は compiled 前提で設計（reference は捨てない）。
- **PDF ビューア差** → pdf.js / Ghostscript / pdfium の 3 系で表示確認。

## 8. 委譲時の最小コンテキスト

- 本ファイル＋ T.88 PDF（ITU 無償）。
- 比較対象 jbig2enc は **動作確認用に別途用意**（出力サイズ比較のみ。ソースは
  clean-room のため読まない）。
- HokusaiPress 側の接続点: `Yomitoku_NDL-OCR-Lite/src/hybrid_ocr/pdf_export.py`
  の `_encode_jbig2` / `_page_pdf_jbig2` / `__post_init__`(PATH チェック)。
- ライセンス: Apache-2.0 もしくは MIT。clean-room・T.88 由来を明記。
