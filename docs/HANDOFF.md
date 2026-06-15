# HokusaiPress 引き継ぎ (Handoff)

最終更新 2026-06-15 / HEAD `f9c20e2` / 86 tests green。
リポジトリ: `C:\Users\user\dev\HokusaiPress`（github.com/science-education/HokusaiPress, GPLv3）。

スキャン PDF を「美しく整い・軽く・全文検索できる」電子書籍 PDF にするバッチアプリ。
バッチ処理 → 低確信ページのみ人間/AI レビュー → 判断蓄積 → 自動化、が運用思想。
**非破壊パラメータ駆動**（ScanTailor 思想）: 各段は画像を書き換えず PageParams だけ記録、
render が元画像から deskew+切抜+解像度統一を 1 アフィンに合成して一度だけワープ。

## 0. すぐ動かす（重要）

OCR/NPU はメイン Python(3.14) では不可（onnxruntime に OpenVINO EP 無し）。
**NPU は sibling の venv を使う**:

```powershell
$py = "C:\Users\user\dev\NDL-OCR-Lite-NPU\.venv-openvino\Scripts\python.exe"  # hokusai editable 導入済
# テスト
& $py -m pytest "C:\Users\user\dev\HokusaiPress" -q
# 実行（NPU並列・キャッシュ）
& $py -m hokusai_press.cli run "<pdf|folder>" --out <dir> --db <db.db> --device npu --openvino-cache-dir "C:\tmp\ov-cache" --workers 2
# 特定頁だけ高速デバッグ（0-based, 真ページ番号保持）
& $py -m hokusai_press.cli run <pdf> --out <dir> --db <db> --pages "9,52,236-237" --device npu
# 段階別タイミング
& $py ... run ... --profile
# レビューUI（落ちにくいよう Start-Process 常駐推奨）
Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList "-m","hokusai_press.cli","review","--db","<db>","--host","127.0.0.1","--port","8000"
```

- モデル: `--model-dir` 省略で `Yomitoku_NDL-OCR-Lite\models` を自動解決。
- **実テストデータ**: `C:\tmp\tmp0613\` に 5 冊（ADF スキャン, 970 頁）: img20260423_0001(264,横書き密)
  / img20260427_0001(176) / img20260427_0010(240,図版多) / img20260430_0002(160) / img20260525_0005(130)。
  ※ 0002/0005 は回転頁混入をユーザーがソース修正済み。
- **長時間バッチ（~60-90分）は run_in_background だとターンをまたいで落ちる**。Start-Process でデタッチ＋
  `-RedirectStandardOutput` のログ＋ temp db で結果回収する。stdout はバッファされ終了時に書かれる。

## 1. このセッションで実装したこと（全コミット済み）

- **NPU 並列 = Plan A**（docs/NPU_PARALLEL_PLAN.md）: マルチプロセスは NPU コンテキスト競合でハング
  → **ThreadPoolExecutor＋単一エンジン共有**（content._get_ocr_engine をロック保護、HybridOCR は
  self不変+ORT thread-safe を確認）。`--device npu --workers 2` で **~1.77倍**・ハングなし実証。
- **deskew 信頼度** = `(best-median)/(median-min)`（図のベースライン不変）、閾値1.5。角度計算は不変。
- **影/縁線除去**（ADF 前提・決定論）:
  - `remove_edge_shadows`(margin.py, 成分ベース) を元画像ワープ前に適用（検出/deskew/OCR/描画 全部影フリー）
  - **margin fill**: render で内容枠の外を白化（ADF の細い縁線を確実に除去）。背景正規化/Sauvola は
    ADF に不適なので不採用（照明むら無し・大濃部を洗い流す害がある、と実測で確認）
- **二値化クランプ**（裏移り・影 penumbra の根本対策）: `ink_threshold = min(Otsu, INK_CEIL=130)`。
  Otsu はブランク頁で高閾値(~250)を出し裏移り/影を黒化する → 絶対上限でクランプ。`binarize_bw`(render)
  を mrc/preview で使用（OCR側 hybrid_ocr.binarize は触らない）。`remove_edge_shadows` と content の
  残差写真検出も同じ ink_threshold に統一。
- **OCRゲート blank 白化**: `PageParams.blank`。`use_ocr かつ 内容領域0 かつ クランプ残インク<0.1%` で
  白紙出力。use_ocr ゲート＋残インク照合で、疎な実1行(p56)や `--no-ocr` を消さない。
- **内容枠** = インク枠 ∪ 検出領域（写真を必ず含む）。面積<12% は検出失敗→全面+MARGIN_NOT_FOUND。
- **判型統一**（全文書一律 max、パリティ別でない）＋ 全四辺 ≥ 出力余白でクランプ（密着回避）＋ no-clip。
- **ノンブル robust**(nombre.py): arabic/漢/ローマ数字、傾き1のオフセット投票で柱・章番号・誤読を排除、
  無番前付け対応、確信度(page_number割当)で在来配置にフォールバック融合。角ノンブル捕捉を実OCRで確認
  （偶=右上/奇=左上, recto/verso）。未読頁の stale ノンブルはクリア。
- **align_margins**: 旧「中央値から±4%＋ノンブル無し」は過剰発火(51件)→ **外れ値的に大きい内容枠
  (>median×1.3)のみ**。判型統一で出力は頑健なので小ばらつきは不問。
- **ocr_low_coverage** 閾値 0.5→0.3 ＋ 図版インク除外。
- CLI: `--pages` / `--profile` / `--workers`(スレッド) / `--openvino-cache-dir` / `--out`フォルダ可 /
  複数・フォルダ・glob 入力。Store は親dir自動作成＋スレッドロック。**store キーは真ページindex**。
- レビューUI: 上部に白黒/グレー/カラー＋キューに戻る(全頁再計算)、描画モード(写真カラー/グレー/内容/
  ノンブル, ドラッグ即追加・ダブルクリック無確認削除・モード別ホバー)、検出領域一覧(クラス別+行削除)、
  出力プレビュー=最終PDF相当、analysis 凡例ボタン。content/nombre は確認ダイアログ→全頁再計算。

## 2. 実測の現状（全5冊 final3.db ※ただし古い）

`C:\tmp\tmp0613\final3.db`（commit a46d4fe 時点）でレビュー 89→**49**（align_margins 修正後, 970頁の~5%）。
**注意: final3.db は二値化クランプ/blank/margin-fill 反映前**。これらで裏移り/影/誤フラグが更に減るはず
→ **最新コードで全5冊を再処理して再確認するのが次の一手**（out-final3/*.pdf は margin fill 済みで端黒px=0 を確認済）。

## 3. 残課題（ユーザーのページ別レビューで確定済み・未実装）

| # | 内容 | 該当ページ |
|---|---|---|
| A | **deskew 過剰フラグ**。img0423 は**横書き**で p207 以外の傾きは適正。信頼度メトリックが密頁で過剰 | 0423 の deskew_low_confidence 18件（p207 のみ実問題） |
| B | **0010 p9 傾き補正失敗**（実際に角度がずれている）＋影 | 0010 p9 |
| C | **検出失敗フォールバックを「全面−細縁帯」に**（content=全面だと margin fill が効かず影が残る） | margin_not_found の一部 |
| D | **gap 判定を同一オフセットクラスタ内に限定**。p261 は誤検知（262 は正しい、p259 の「16」は別クラスタ） | 0423 p261, 0010 p9 |
| E | **0423 p29 網掛け**: 紙面四辺の網を「次の領域まで」背景除去＋ノンブル残存 | 0423 p29 |

補足の設計メモ:
- C/D は小修正。D は nombre._gap_warnings で各ページの「割当オフセット」を持たせ、同一オフセット連続のみ比較。
- A は「角度計算は触らず信頼度だけ」がユーザー方針。横書き/縦書きで投影軸が違う点に注意（密な横書きで
  信頼度が下がりやすい）。実データで分布を測って閾値/メトリック再校正（deskew のときと同じ進め方）。
- 二大頻出だった「裏移り・左右影」は二値化クランプ＋OCRゲート blank＋margin fill で対応済み。再処理で
  実残量を確認すること。

## 4. ファイル早見

- `pipeline.py`: analyze_document（deskew→content→margin→nombre→align/normalize）, run, rebuild,
  recompute_margins, blank判定, MIN_CONTENT_AREA_FRAC/BLANK_INK_FRAC
- `geometry/margin.py`: find_content_box, remove_edge_shadows, ink_threshold(INK_CEIL), normalize_margins
  (判型統一・融合配置・≥余白クランプ), align_margins
- `geometry/deskew.py`: find_skew, 信頼度メトリック, GOOD_CONFIDENCE=1.5
- `render.py`: render_page_image(blank白化/margin fill/影除去), binarize_bw, render_output_preview
- `content.py`: analyze(OCR+残差写真, ink_threshold統一), LOW_COVERAGE_FRAC=0.3, エンジン構築失敗は例外送出
- `nombre.py`: parse_numeral/parse_roman, resolve(オフセット投票), _gap_warnings
- `mrc.py`: MRC合成（binarize_bw使用）, `webui/app.py`: レビューUI, `cli.py`: 全サブコマンド
