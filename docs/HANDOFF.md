# HokusaiPress 引き継ぎ (Handoff)

最終更新 2026-06-23 / HEAD `ea892ef` (branch `param-profile`) / 170 tests green。
リポジトリ: `C:\Users\user\dev\HokusaiPress`（github.com/science-education/HokusaiPress, GPLv3）。
**PR**: [#1](https://github.com/science-education/HokusaiPress/pull/1)（`param-profile` → `main`, CI green）。

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
- **二値化（裏移り・影 penumbra 対策, 2026-06-15 再校正）**: `ink_threshold` は **Otsu をそのまま使う**。
  当初の `min(Otsu, INK_CEIL=130)` は誤りだった — このADFの実文字は ~130-218 のグレー帯にあり(純黒<100は
  ごく僅か)、130 にクランプすると内容頁でインクの **38-53%** を白に落とし**全頁かすれ＋影成分を見落とす**。
  Otsu が信用できないのは degenerate な near-blank 頁(裏移りのみで実インク無し→Otsu が ~250 へ暴走)だけ。
  そこで **2信号で degenerate を判定**: `Otsu > INK_VALLEY_MAX(225)` かつ `<INK_FLOOR(110) の画素 < 0.1%`
  のときだけ閾値を `INK_FLOOR=110` に落として裏移りを棄却（blank ゲートが回収）。実測970頁: 実頁は Otsu≤218
  で純黒画素≥1%、裏移り頁(0423 p2 / 0005 p2)のみ Otsu~250・純黒0.000% という明確な2信号分離。`binarize_bw`
  (render)を mrc/preview で使用（OCR側 hybrid_ocr.binarize は触らない）。`remove_edge_shadows` と content の
  残差写真検出も同じ ink_threshold に統一（Otsu に戻したことで中間グレーの影も成分として拾える）。
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
| C | **検出失敗フォールバックを「全面−細縁帯」に**（content=全面だと margin fill が効かず影が残る）。**near-blank 頁で分断された綴じ目縦線が残る**（remove_edge_shadows の単一成分≥半辺ルールが分断線を拾えない。旧クランプ@130 は本文ごと消すことで偶然隠していた→Otsu 復帰で顕在化）。要：極端外側帯の列密度フィル or margin fill の確実化 | margin_not_found の一部, 0427_0001 disp13 |
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

## 5. 2026-06-19 追記: tint上の白抜きtextは黒二値化しない

対象: `C:\tmp\improvements_review\abc_visual\quality_FINAL4.pdf` / `badge_after_despeckle.png` で残っていた
「新学習指導要領…」白抜きタイトル内の黒部と、`コラム` バッジ内の黒クラスタ。

実装方針:
- 問題は最終 despeckle 不足ではなく、tint zone 内で白抜き文字を「黒い bilevel text」と誤分類していたこと。
- OCR の `lines` は検索用 text として残す一方、見た目が白抜きの text box は bilevel 黒インク対象から外す。
- `mrc._knockout_text_mask()` が tint zone と交差する OCR text box を見て、
  bright fraction >= 0.02 かつ dark fraction <= 0.12 なら knockout text と判定し、その box を
  `_tint_ink_mask(..., knockout_text_mask)` で黒化禁止にする。
- 併せて `_local_tint_background()` は白抜き/黒ストロークを tint median で埋めてから median filter する。
  これで白抜き文字が local background を押し上げて周囲の粒子を黒化する副作用を抑える。
- 残る小さな白抜き近傍ノイズ向けに component veto は維持するが、主役ではない。

検証:
- `tests/test_mrc.py` に synthetic case 追加:
  `test_tint_ink_mask_vetoes_knockout_noise_but_keeps_real_strokes`,
  `test_tint_ink_mask_excludes_ocr_knockout_text_box`。
- 実行: `& C:\Users\user\dev\NDL-OCR-Lite-NPU\.venv-openvino\Scripts\python.exe -m pytest C:\Users\user\dev\HokusaiPress\tests\test_mrc.py -q`
  → 7 passed。
- 実ページ: `& ...python.exe C:\tmp\improvements_review\abc_visual\_quality_check.py`
  → `tint_render_quality found 0 problem tiles`。
- 目視 crop: `C:\tmp\improvements_review\abc_visual\codex_after_badge.png`,
  `codex_after_title.png`, `codex_after_overview_top.png`。

## 6. 2026-06-19 追記: p29 右端の黒塊は内部tint濃淡境界で発生

対象: `C:\Users\user\Downloads\クリップボード_06-19-2026_03.png` の中央右に残った黒塊。

診断:
- 該当小窓を各段階で保存: `C:\tmp\improvements_review\abc_visual\edge_debug\`。
- 重要ファイル:
  - `01_render_page_image_gray.png`: 元レンダー。黒塊なし。
  - `18_zone1_overlay_raw.png`: tint overlay。黒塊なし。
  - `14_zone1_ink_before_edge.png`: ここで黒塊が初出。
  - `20_binary_after_tint.png` / `51_pdf_rendered_gray.png`: ink由来の黒塊が残存。
- `vecfills 0` だったため vector fill 経路ではない。
- relevant zone は `(122,117)-(4123,6040)` の大きな poly zone。問題箇所は zone 外縁ではなく、濃い網掛け矩形と淡色帯が接する内部角。
  そのため既存の外縁用 `_tint_overlay_edge_dirt_mask()` は `edge_dirt crop px 0` で反応しない。
- 主成分は page bbox `(4104,1013)-(4119,1026)`, area 102, pixel min/mean/max `103/125.6/143`,
  local_bg mean 約183。真の黒文字のような暗い芯を持たず、濃淡境界近傍の網点が連結したもの。

実装:
- `_tint_ink_mask()` の連結成分後処理に、内部tint濃淡境界ノイズの veto を追加。
- 条件は「小成分」「暗い芯がない」「周辺8pxの濃度レンジが大きい」場合だけ除外。
  本物の黒ストロークは暗い芯を持つので残る。
- 外縁処理を別経路で増やすのではなく、既存の tint ink 判定の後処理として整理。

検証:
- `tests/test_mrc.py` に `test_tint_ink_mask_removes_tone_edge_halftone_cluster` を追加。
- 修正後の実ページ局所:
  - `80_zone1_ink_after_fix.png`: 主黒塊は消え、5pxの孤立点のみ。
  - `82_binary_after_despeckle_fix.png`: 既存 despeckle 後は黒ピクセル0。
- 実行: `python -m pytest tests/test_mrc.py` → 12 passed。

## 7. 2026-06-19 追記: 単体PNGをHokusaiPressフル経路で処理

ユーザー依頼: `C:\tmp\improvements_review\abc_visual\b_p29_before_gray.png` を、OCRなどHokusaiPressの機能をフルに使って処理。

実行コマンド:

```powershell
python -m hokusai_press.cli run "C:\tmp\improvements_review\abc_visual\b_p29_before_gray.png" `
  --out "C:\tmp\improvements_review\abc_visual\b_p29_hokusai_full.pdf" `
  --db "C:\tmp\improvements_review\abc_visual\b_p29_hokusai_full.db" `
  --device auto `
  --openvino-cache-dir "C:\tmp\improvements_review\abc_visual\openvino_cache" `
  --profile
```

結果:
- 出力PDF: `C:\tmp\improvements_review\abc_visual\b_p29_hokusai_full.pdf`
- DB: `C:\tmp\improvements_review\abc_visual\b_p29_hokusai_full.db`
- 確認レンダー: `C:\tmp\improvements_review\abc_visual\b_p29_hokusai_full_render.png`
- タイトル周辺crop: `C:\tmp\improvements_review\abc_visual\b_p29_hokusai_full_title_crop.png`
- `hokusai_press.cli queue --db ...` → `review queue is empty`
- 処理時間: TPB 56.5s。内訳は render+pdf 31.1s, OCR 24.2s。
- ログ末尾に `No accelerator EP available; running on CPU only.` と出たため、この実行はCPU OCR。

目視確認:
- `b_p29_hokusai_full_title_crop.png` では、以前問題にしていたタイトル右側/右下の真っ黒な塊は見えない。
- 白抜きタイトルはグレーtint overlay上に残り、bilevel黒インク化していない。

注意:
- このPNGは既にグレー化/ページ化された単体入力。原PDF+DBの既存ページ処理とは入力条件が違う。
- 単体PNGなのでOCR text boxはこの実行で再作成される。以前の `final2.db` の行情報とは一致しない。
- 後続AIが比較するときは、`b_p29_before_gray.png` と上記 `b_p29_hokusai_full*.png/pdf` を同じ表示倍率で見ること。

## 8. 2026-06-19 追記: tintベクター化は断片化時に安全フォールバック

対象: `src/hokusai_press/tint_zone.py` の `try_vectorize_zone()`。

背景:
- 旧ベクター実装は `img20260423_0001.pdf` page_index=29 で、全ラスター 939KB 相当から旧vector 993KB 相当へ増加していた。
- 原因は、平網のスキャンノイズ/粒状感が k-means 後に塩こしょう状の微小連結成分になり、数百から数千のPDFパスとして出力されること。

実装:
- 各クラスタの `color_mask` に 7x7 楕円カーネルの close/open を適用。
- 平滑化で未割当になった描画画素は、残ったクラスタマスクへの距離で近傍優勢色に再割当。
- 最小成分面積を `max(64px, zone_area*0.0001)`、上限1000pxへ引き上げ。
- 採用前に、総連結成分数 50 以下、総頂点数 500 以下、平滑化後も描画画素の95%以上が±20階調内、を満たす場合だけ `zone.vectorize_success=True`。
- 予算超過や品質不足時は `False` を返し、既存JPEG tint overlayへフォールバック。
- 2026-06-19追加修正: 実ページPNGで検証中、巨大zoneの全画素k-means/輪郭化がタイムアウトしたため、
  k-means入力は最大200,000pxの決定的サンプルに制限し、描画対象が750,000pxを超えるzoneは即フォールバック。
  ベクター化は中小サイズの滑らかなtint専用とする。

検証:
- 追加テスト: `tests/test_mrc.py::test_tint_vectorization_decision` に、2色でも塩こしょう状に断片化した入力は `False` になるケースを追加。
- 実ページ比較:
  - 入力: `C:\tmp\tmp0613\final2.db` / `C:\tmp\tmp0613\img20260423_0001.pdf` / page_index=29。
  - 出力: `C:\tmp\tmp0613\vector_tint_check\p29_raster_baseline.pdf` と `p29_vector_safe.pdf`。
  - raster baseline: 961,865 bytes = 939.3 KiB。
  - 旧vector実装: 993 KiB 相当（ユーザー報告値、+5.73%）。
  - 新実装: 961,865 bytes = 939.3 KiB、delta 0 bytes / 0.0%。
  - `zones 2`, `vectorized_zones 0`, `vector_fills 0`。このページでは安全側に倒して全ラスターへフォールバック。
  - `tint_render_quality(..., deviation_threshold=40.0)` → `quality_issues 0`。
- 全テスト:
  `& C:\Users\user\dev\NDL-OCR-Lite-NPU\.venv-openvino\Scripts\python.exe -m pytest -q`
  → 146 passed, 2 warnings。
- 追加検証:
  - `C:\tmp\improvements_review\abc_visual\vector_verify\smooth_vector_tint.pdf`
    は `zone.vectorize_success=True`, `fills=3`, PDF画像filterはCCITTのみ、DCT/JPEGなし。
  - `C:\tmp\improvements_review\abc_visual\b_p29_before_gray.png` をベクター化単体検証:
    panels 3, zones 2。badge zone `(429,615)-(889,1076)` は `pixels=154834`, `ok=False`。
    page zone `(63,64)-(4234,6239)` は `pixels=5923929`, `ok=False`。
    合計 `vectorized_zones 0`, `total_sec 1.72`。

2026-06-20 追加:
- ユーザー方針: 網点は故意の絵柄ではなく印刷/光学制約によることが多く、本質的には単色面として扱える場合がある。
- `try_vectorize_zone()` に flat-tone 経路を追加。通常の少数色ポリゴン化より前に、
  大きなhalftone面を低周波クラスタで代表色ポリゴン化する。
- 判定:
  - 対象が100,000px以上ならflat-tone候補。
  - 低解像度（最大辺900px）で1-4階調k-means。
  - 代表色との差が25階調以内の画素が70%以上あれば候補。
  - タイル中央値IQRが28以下の成分だけ採用し、写真/グラデーションを避ける。
  - 輪郭抽出も低解像度マスク側で行い、スキャン粒状境界で頂点が爆発しないようにする。
  - 元解像度の白抜き/紙白成分（gray>=245）は穴として戻す。穴を戻すと頂点予算を超える場合はフォールバック。
- p29調査:
  - 基準2（巨大zone即フォールバック）と基準4（平滑化後95%品質）を止めると、
    badge zoneは `components=20`, `vertices≈347` でベクター化可能。
    page zoneは `vertex_budget_exceeded (562>500)` で落ちた。
  - 低周波輪郭化を入れるとpage zoneも一時は `fills=19`, `vertices=914` で通ったが、
    PDF目視で白抜きタイトルが崩れた。
  - そのため元解像度の白抜き穴を戻す処理を追加。結果、`b_p29_before_gray.png` の2 zoneは
    白抜き保護の複雑度が高く、どちらも安全フォールバック。
- 追加テスト:
  `tests/test_mrc.py::test_tint_vectorization_flat_large_halftone_with_simple_knockout`
  で、900x900の大きな網点面 + 単純な白抜き穴はflat-toneベクター化され、穴も保持されることを確認。
- 全テスト: `python -m pytest` → 147 passed, 1 warning。

2026-06-20 追加2:
- ユーザー方針: 白抜きでも黒字でも、OCR座標でテキストを把握し、テキスト領域は最小限保護して、
  残りの単色領域を大きなポリゴンで表す。
- 実装:
  - `MrcPageBuilder.add_page()` で tint zone ごとに `_ocr_text_box_mask()` を作り、
    `try_vectorize_zone(..., protect_mask=text_protect)` へ渡す。
  - `_ocr_text_box_mask()` はOCR box全体ではなく、白抜き文字らしいbox内の明るい画素だけを抽出して少し膨張する。
    黒い通常文字は既存のbase/bilevel ink層に任せる。
  - `try_vectorize_zone()` / flat-tone 経路では `protect_mask` を塗り対象から除外し、
    `_protected_hole_contours()` で保護画素をベクター塗りの穴にする。
  - 日本語白抜き文字の穴は頂点が増えるため、flat-toneの頂点上限を `1,200` から `6,000` に上げた。
- 捨てた中間案:
  - OCR box全体をラスタパッチで戻す案は、タイトル帯に大きな矩形差が見えた。
  - 明るい/暗い文字画素をまとめて小ラスタパッチで戻す案は、タイトルの「新学習指導要領...」付近に黒い箱状の汚れが出た。
  - 結論: このケースではテキスト保護を小画像で戻すより、白抜き文字をポリゴン穴として扱うほうがよい。
- p29再検証:
  - 入力: `C:\tmp\improvements_review\abc_visual\b_p29_before_gray.png`
  - 出力: `C:\tmp\improvements_review\abc_visual\flat_hole_vector_rebuild.pdf`
  - レンダー: `C:\tmp\improvements_review\abc_visual\flat_hole_vector_rebuild_render.png`
  - タイトル確認: `C:\tmp\improvements_review\abc_visual\flat_hole_vector_rebuild_title_tight.png`
  - PDFサイズ: `530,643 bytes`。比較用の全ラスター寄り `b_p29_hokusai_full.pdf` は `1,365,991 bytes`。
  - 目視: 以前の黒い矩形/塊は消えた。白抜き文字輪郭には網点由来の粗さが少し残る。
- テスト:
  - `python -m pytest tests/test_mrc.py::test_tint_vectorization_flat_large_halftone_with_simple_knockout tests/test_mrc.py::test_tint_vectorization_uses_holes_for_protected_text_pixels tests/test_mrc.py::test_mrc_with_vectorized_tints`
    → 3 passed。
  - `python -m pytest` → 148 passed, 1 warning。

2026-06-20 追加3:
- ユーザー判断: PDF図形要素への還元は、継ぎ目が目立つため中止。
- 実装:
  - `MrcPageBuilder.add_page()` の tint zone 経路から `try_vectorize_zone()` 呼び出しを外した。
  - tint は従来どおり、300dpi相当のグレー raster overlay + Multiply blend + contour clip として出す。
  - `tint_zone.py` のベクター化実験コードは残っているが、PDF生成経路では使われない。
- 再検証:
  - 入力: `C:\tmp\improvements_review\abc_visual\b_p29_before_gray.png`
  - 出力: `C:\tmp\improvements_review\abc_visual\raster_no_vector_rebuild.pdf`
  - PDFサイズ: `1,365,991 bytes`
  - image filters: `CCITTFaxDecode` + `DCTDecode` x3。つまり現在のtint overlayはJPEG。
  - タイトル確認: `C:\tmp\improvements_review\abc_visual\raster_no_vector_rebuild_title_tight.png`
- 減色 + LZW/可逆圧縮の調査:
  - 実測フォルダ: `C:\tmp\improvements_review\abc_visual\posterize_lzw_probe\`
  - 結果JSON: `posterize_lzw_probe\results.json`
  - 比較画像: `posterize_lzw_probe\title_posterize_compare.png`
  - 対象は `b_p29_before_gray.png` の検出tint zoneをMRC同様に300dpi相当へ縮小したもの。
  - 大きいpage zoneの主な結果:
    - original: JPEG q85 `881,008`, TIFF LZW `2,362,766`, PNG/Flate `1,844,750`。無減色の可逆はJPEGより大きい。
    - k-means 4色: JPEG q85 `978,128`, TIFF LZW `420,252`, PDF化TIFF `518,739`, MAE `3.423`, p95 `15`。
    - k-means 8色: JPEG q85 `931,488`, TIFF LZW `650,836`, PDF化TIFF `858,476`, MAE `1.539`, p95 `8`。
    - k-means 16色: JPEG q85 `893,403`, TIFF LZW `981,270`, PDF化TIFF `1,269,943`, MAE `0.776`, p95 `4`。
  - 小さいbadge zoneでは、k-means 4色/8色とも単体LZWはJPEG以下だが、差は小さい。
  - `img2pdf` にTIFF LZWを渡してPDF化すると、PDF image filterは `/LZWDecode` ではなく `/FlateDecode` になった。
    そのため現依存で素直に実装するなら、実名はLZWではなく「posterized Flate/PNG系 overlay」になる可能性が高い。
- 暫定評価:
  - LZW/Flateは減色なしでは不利。
  - k4は大きく縮むが、タイトル帯の階調段差が目立つ。
  - k8は見た目とサイズの折衷候補。PDF化後のサイズ差はJPEG比で小さいが、TIFF LZW単体では明確に小さい。
  - 次に実装するなら、tint overlay限定で `k=8` 前後のposterizeを行い、PDF内にIndexed/DeviceGray + FlateDecodeで直接埋める経路を作るのが現実的。

## 9. 2026-06-19 追記: JBIG2をWSL内jbig2encで使う手順 / 実導入済み

目的:
- HokusaiPress の `RenderSettings.bilevel_codec` は `g4 | jbig2`。
- 実際のJBIG2符号化は `hybrid_ocr.pdf_export` が担当し、`shutil.which("jbig2")` と
  `subprocess.run(["jbig2", "-p", png])` を使う。
- したがってWindows側PATHに `jbig2` というコマンドが見えればよい。WSL内で `jbig2enc` をビルドし、
  Windows側にラッパーを置く。
- `-p` は generic coding。symbol modeを使わないので、JBIG2の文字置換リスクを避ける方針。

実施済み:
- WSL2 Ubuntu-24.04 内で `jbig2enc 0.31` を `/usr/local/bin/jbig2` にインストール済み。
- Windows側ラッパー:
  - `C:\tools\jbig2enc\jbig2.exe`。Goで作成した実行ファイル。Windowsパス引数を `wslpath` で変換し、
    `wsl.exe -d Ubuntu-24.04 -e jbig2 ...` を実行する。stdout/stderrはバイナリのまま通す。
  - `C:\tools\jbig2enc\jbig2-wrapper.go` にソースを保存。
  - `C:\tools\jbig2enc\jbig2.cmd` も残っているが、Python `subprocess.run(["jbig2", ...])` では `.cmd` は
    直接起動できないため、実運用上は `jbig2.exe` が必要。
- User PATH に `C:\tools\jbig2enc` を追加済み。現在の長寿命プロセスではPATH再読込が必要な場合がある。

再現手順:

WSL側ビルド:

```bash
sudo apt update
sudo apt install -y git build-essential autoconf automake libtool pkg-config libleptonica-dev
mkdir -p ~/src
cd ~/src
git clone https://github.com/agl/jbig2enc.git
cd jbig2enc
./autogen.sh
./configure
make -j"$(nproc)"
sudo make install
command -v jbig2
jbig2 --version || true
```

Windows側ラッパー:

`jbig2.cmd` ではPythonから直接起動できないため、Go等で `jbig2.exe` を作る。現物は
`C:\tools\jbig2enc\jbig2-wrapper.go` を参照。

PATH追加と確認:

```powershell
$env:Path = "C:\tools\jbig2enc;$env:Path"
[Environment]::SetEnvironmentVariable(
  "Path",
  "C:\tools\jbig2enc;" + [Environment]::GetEnvironmentVariable("Path", "User"),
  "User"
)
Get-Command jbig2
```

動作確認:

```powershell
@'
import cv2, numpy as np, subprocess, shutil, tempfile, os
print("which", shutil.which("jbig2"))
img = np.full((200, 200), 255, np.uint8)
cv2.putText(img, "TEST", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 5, cv2.LINE_AA)
_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "page.png")
    cv2.imwrite(p, bw)
    r = subprocess.run(["jbig2", "-p", p], capture_output=True, text=False)
    print("returncode", r.returncode, "stdout bytes", len(r.stdout), "stderr", r.stderr[:200])
    assert r.returncode == 0
    assert len(r.stdout) > 0
'@ | python -
```

確認済み:
- `python` から `subprocess.run(["jbig2", "-p", png], capture_output=True)` → returncode 0,
  stdout 209 bytes。
- `MrcPageBuilder(compress="jbig2")` で
  `C:\tmp\improvements_review\abc_visual\jbig2_verify\jbig2_mrc_test.pdf` を生成。
  PDF image filter は `/JBIG2Decode`。

HokusaiPressでの利用:
- DB/設定上は `Document.render.bilevel_codec = "jbig2"` を使う。
- CLIに直接 `--pdf-compress` は無い。現状は設定/DB経由または必要ならCLIオプション追加が別作業。
- `jbig2` がPATHに無い状態で `jbig2` codecを選ぶと、`hybrid_ocr.pdf_export.SearchablePdfBuilder.__post_init__`
  由来のRuntimeErrorになる。

注意:
- WSLが初回起動で遅い場合、最初の `wsl.exe` 呼び出しだけ時間がかかる。
- ラッパーはASCIIで保存する。
- `jbig2enc` のlossless generic mode前提。`-s` などsymbol coding系オプションは使わない。

## 10. 2026-06-20 追記: tint overlay に k4 posterize + FlateDecode 経路を追加

対象:
- `src/hokusai_press/mrc.py`
- `tests/test_mrc.py`

実装:
- tint overlay のJPEG生成前に `_try_posterized_tint_overlay_pdf()` を追加。
- 判定と処理順:
  - `in_shape & ~ink & overlay < 255` だけを量子化対象にする。
  - 量子化対象外の紙白/形状外は255のまま、ink画素は元overlay値のまま保持する。
  - `_smooth_tint_for_posterize()` でメディアン平滑化した下地を作る。
  - 写真判定は既存 `region_class.classify_patch()` を使う。大きな複合tint zone全体を一括判定すると複数平坦トーンの分散でphoto扱いになりやすいため、平滑化後の有効画素を256pxタイル単位で判定し、photoタイルが過半ならJPEGへフォールバックする。
  - `cv2.kmeans(..., K=4)` で代表階調を作る。
  - クラスタ学習は平滑化画像で行い、最終ラベルは元overlay画素に近い代表値へ割り当てる。これは濃淡境界がメディアン平滑化で明るいクラスタに吸われるのを避けるため。
  - 品質ガード: `mae <= 10.0`, `p95 <= 28.0`, `abs(diff) > 32` の画素率 `<= 0.03`。外れた場合は従来JPEG。
  - 成功時は `_encode_gray_flate_page_pdf()` で8bit DeviceGray画像を `/FlateDecode` streamとして直接PDF化する。
- `_add_tint_overlays()` のMultiplyブレンド、輪郭/穴クリップ、配置ロジックは従来どおり。

追加/更新テスト:
- `test_tint_posterized_flate_beats_jpeg_on_flat_halftone`
  - 平坦2トーン + 粒状ノイズの合成tintでk4+Flateが採用され、JPEG PDFより小さいことを確認。
- `test_tint_posterize_skips_photo_like_overlay`
  - 連続階調の合成画像で `classify_patch(...) == "photo"` となり、posterize経路が `None` を返すことを確認。
- `test_tint_posterize_preserves_ink_and_paper_pixels`
  - ink画素と白紙/白抜き画素がposterize後のFlate画像でも保持されることを確認。
- 既存 `test_mrc_keeps_tints_as_raster_overlays` は、ベクター化されないことを維持しつつ、画像filterはDCTまたはFlateのラスターoverlayとして確認する形に更新。

検証:
- `& C:\Users\user\dev\NDL-OCR-Lite-NPU\.venv-openvino\Scripts\python.exe -m pytest tests\test_mrc.py -q`
  - 代表値再割当前: `19 passed, 1 warning`。
  - その後、p29右端のtoo_light境界タイル対策として「最終ラベルを元overlay画素に近い代表値へ割当」に変更。
  - 変更後に再実行しようとしたが、PowerShell/プロセス起動が `-1073741502` で失敗し、`Get-Date` や `python -c "print('ok')"` も同じ終了コードになったため、この最終差分後のpytest/full pytestは未完了。
- p29実測（最終ラベル割当前の計測。境界タイル対策後はシェル障害で未再計測）:
  - 入力: `C:\tmp\tmp0613\final2.db` / `img20260423_0001.pdf` / `page_index=29`。
  - 出力: `C:\tmp\improvements_review\abc_visual\posterized_flate_verify\`
  - JPEG baseline: `p29_jpeg_baseline.pdf` = `961,865 bytes`。
  - k4+Flate: `p29_k4_flate.pdf` = `193,098 bytes`。
  - delta: `-768,767 bytes`。
  - filters baseline: `CCITTFaxDecode + DCTDecode x2`。
  - filters new: `CCITTFaxDecode + DCTDecode + FlateDecode`。
  - zones: 2。
  - zone0（badge）は `jpeg_fallback`。
  - zone1（大ゾーン）は `k4_flate`, `class=solid_fill`, `mae=9.42`, `p95=26`, `bad_frac=0.0298`, centers `[134, 163, 187, 192]`。
  - `tint_render_quality(..., deviation_threshold=40.0)` は境界タイル1件（`too_light`, deviation約40.8）を検出したため、上記の最終ラベル割当修正を追加した。
  - 保存crop: `p29_k4_flate_title_crop.png`, `p29_k4_flate_badge_crop.png`。

### 10b. 2026-06-20 追記: 最終検証完走 + コミット済み（後続セッション向け）

Codexのシェル障害(`-1073741502`)で未完だった検証を、別セッションで完走させた。

- 原因: 多数の孤児プロセス(codex.exe等)残留によるリソース枯渇。`taskkill /F /IM codex.exe` で解消後は正常動作。
- **full pytest: `151 passed, 2 warnings`**（`tests/` 全体）。
- **p29再計測（最終ラベル割当修正後の現行コード）**:
  - スクリプトはインラインで再実行（`render_page_image`→`build_pdf`→pikepdfでfilter確認→pdfiumでレンダリングし`tint_render_quality`）。
  - 出力: `C:\tmp\improvements_review\abc_visual\k4flate_FINAL.pdf`。
  - **サイズ: `961,865 -> 390,823 bytes`（約59%削減）**。
    - ※Codex報告の193,098 bytesより大きいが、これは最終ラベル割当修正(境界タイル対策)を含む現行コードでの実測値。193KB版は対策前の計測なので、391KBが正。
  - filters: `CCITTFaxDecode + DCTDecode + FlateDecode`（意図通り）。
  - **`tint_render_quality(..., deviation_threshold=40.0)` → 問題タイル0件**（境界タイル対策が効いている）。
  - 目視crop: `k4_title.png`(タイトル帯=k4+Flate), `k4_badge.png`(コラム=JPEGフォールバック), `k4_fullpage.png`(全体)。いずれも文字滲み・黒塊なし。
- **コミット済み**:
  - HokusaiPress(ブランチ `param-profile`): `4137ba7` "Compress tint overlays via photo-aware k4 posterize + FlateDecode"
  - Yomitoku_NDL-OCR-Lite(ブランチ `main`): `b44b07f` "Fix _page_pdf_jbig2 page construction for newer pikepdf"

### 残課題・次セッションの注意点
- **JBIG2は導入済みだが既定では未使用**: `MrcPageBuilder(compress=...)` のデフォルトは `g4`。JBIG2を使うには `compress="jbig2"` を渡す必要がある。`jbig2`バイナリはWSLラッパー(`%LOCALAPPDATA%\jbig2-wrapper\jbig2.cmd`)経由で、**新しいシェルのPATHにwrapperディレクトリが通っていることが前提**。CI/別マシンでは未導入なので、JBIG2をデフォルト化する場合はフォールバック設計が要る。
- **k4 posterizeのパラメータ**は p29 一枚で調整した値（`mae<=10`, `p95<=28`, `bad_frac<=0.03`, K=4）。他ページ・他冊子での汎化は未検証。写真誤判定が起きると破綻するので、複数ページでの回帰確認が望ましい。
- これらk4関連定数・関数は `src/hokusai_press/mrc.py`(`_try_posterized_tint_overlay_pdf`, `_smooth_tint_for_posterize`, `_encode_gray_flate_page_pdf` 等)にある。

## 11. 2026-06-23 追記: margin/影検出の再設計、判型統一の強制、NPU並列化（PR #1）

ユーザーが実データ（`C:\tmp\tmp0613\img20260427_0001.pdf` 中心、一部 `img20260423_0001.pdf` /
`img20260430_0002.pdf` で横展開確認）を1ページずつ目視レビューし、「影が消えていない」「コンテンツ枠が
おかしい」「判型が統一されていない」等を1つずつ実データで検証→修正、を繰り返したセッション。
**6+1コミット、`param-profile` ブランチに積んで [PR #1](https://github.com/science-education/HokusaiPress/pull/1) 作成、CI green。**

### 影検出: run-length方式 → 形状ベース方式に再設計

`region_shadow_mask`(margin.py) の旧方式（テキスト枠外で「ページ高の25%以上」連続して暗い列を起点に
±halo狭め白化）は、**短い影・先細りする影・ワーンアウトしたADFローラーの斑点状（断続）の影**を取り逃す。
新方式は形状で判定: テキスト枠外の連結成分が `height >= SHADOW_MIN_HEIGHT_PX(15)` かつ
`height/width >= SHADOW_ASPECT_MIN(6)`（細長い）なら影として丸ごと白化。判定前に縦方向だけ小さく
dilateして数px断片を結合（疎なマージン文字のdot同士までは結合しない距離）。halo は 3→8px に拡大
（影が両端でわずかに蛇行するため）。**TEXT領域が1つも無いページ（白紙寄り）は、旧コードは何もしない
仕様だったが、新コードはページ全幅をスキャン対象にする**（テキストが無いページは消す対象を誤検知する
リスクも無い）。

### `find_content_box` の二値化を Otsu直結から ink_threshold() に変更

`_deskewed_binary`(margin.py) が `cv2.threshold(..., OTSU)` を直接使っていたため、**ほぼ白紙のページで
Otsuが暴走**（ノイズと背景を分離してしまい250前後の異常閾値になる）し、薄いアンチエイリアスノイズを
コンテンツとして誤検出 → そのページのcontent boxが異常に大きくなっていた。`ink_threshold()`（既存の
2信号degenerate判定）に統一して解消。

### コンテンツ枠は「regionsベースのみ」を信頼。ink-bboxは regions が空の時だけのフォールバック

`pipeline.py: compute_margin()`。以前は `find_content_box` のインク連結成分スキャンの結果と
regions（OCR/レイアウト検出）の和集合をcontent boxにしていたが、**影検出が取り逃した断片や薄いゴミが
ink-bbox側に残っていると、それがcontent boxを膨らませてレンダリング時のmargin-fillを無効化する**
（枠の外側しか白化できないので、枠自体に影が取り込まれていると消せない）。regionsが1つでもあれば
それだけでcontent boxを決め、ink-bboxは無視する。**regionsが空（＝何も検出されていない）場合のみ**
`MIN_CONTENT_AREA_FRAC` 失敗時の全面フォールバック（confidence=0）を使う。

注意点（要再発防止）: 「regionが1つでもあれば信頼する」を入れたら、`原理編`/`実践編`等の**短い部タイトル
1行だけのページが、それまでは「検出失敗」とみなされ全面フォールバック→正規化で中央寄せされていたのに、
今回の修正でそのregion box単体がcontent boxになり、しかも以前の「白紙ページ用センタリング」のままだと
**右寄り・上寄りの本物のタイトル位置がクロップ範囲外に出て完全に消える**事故があった
（後述の「proportional position」修正で解消）。

### 判型統一（page-format uniformity）を「床上げ」から「縮小フィット」に変更

旧 `crop_size()`（normalize_margins, margin.py）は「統一サイズ」と「自ページの内容サイズ」の大きい方を
採用 → 章扉の円形フルブリード図版1ページのために**そのページだけ物理サイズが大きくなる**事故があった
（ユーザー指摘: 「判型統一は基本。観音開きが唯一の例外」）。

修正方針: **クロップ領域（元画像から取得するSOURCE範囲）と出力サイズ（最終ページの物理サイズ）を分離**。
- `crop_size()`: 引き続き自ページの内容に床上げ（クリップ防止、変更なし）。
- 新規 `target_size()`: 書籍全体で固定の統一出力サイズ（P97.5パーセンタイル基準、変更なし）。
- `Margin.target_w/target_h` に保存（`model.py`）。
- `render.py: compose_transform()` で、crop領域が target を超える場合は **アスペクト比を保ったまま
  縮小**し、target_w×target_h の固定キャンバスの中央に配置（クリップなし、歪みなし）。
- 縮小が起きたページは `Flag.CONTENT_SCALED_DOWN` でレビューフラグ。
- 実データ確認: `img20260430_0002.pdf` の章扉ページ（円形図版、idx=6/58）で実証。全ページ完全に同一
  出力サイズ（3218×4827px）になり、円形図版も歪まず縮小収納された。

副産物のバグ修正: `render.py` の `photo_boxes` が出力canvas外/縮小後ほぼ0pxになる場合に
img2pdf/pikepdfが `Page size must be between 3 and 14400 PDF units` で落ちる問題を、クリップ＋
スキップで解消（`mrc.py` 側にも同様のガード追加）。

### コンテンツの「元のページ上の相対位置」を保持する配置（dead-centerをやめる）

`baseline()`(normalize_margins, margin.py)。ノンブルで位置確定できないページを「統一クロップの中央に
強制配置」していたのを、**元ページでの相対位置（横%・縦%）を保ったまま新クロップに配置**するよう変更。
理由は上述の「部タイトルページが中央寄せで消える」事故。`Margin.page_w/page_h`（元の deskew後ページ
サイズ, `model.py`）を追加してこのfractionを計算。全面フォールバック（content=全面）の場合は元々
fx=fy=0.5になるので、分岐なしで両方のケースをカバーする。実測: idx=12「原理編」が fx=94.5%(右端寄り)・
fy=27.3%(上から3割)で検出され、ユーザーの目視推定とほぼ一致。

### ノンブル: 上端基準→下端基準アンカー、かつ「数値が信用できなくても位置だけ使う」復帰パス追加

- `normalize_margins` のノンブル縦アンカーを `nombre_box.y0`(上端)から `nombre_box.y1`(下端)に変更。
  単桁「9」と3桁「126」は字形bboxの高さが違うが、同じベースラインに乗るため下端基準が正しい。
- `nombre.py: _assign_with_position_model()` に **pass 3**を追加: OCR数値が信用できない
  （前付けの短い連番で `_primary_clusters` の閾値に届かない、または `parse_numeral("00")==0` で
  `v>0` フィルタに弾かれ候補にすらならない）ページでも、**ノンブル帯の領域が位置モデル上の期待座標に
  あれば**、page_numberは付与せず`nombre_box`だけ採用してマージンアンカーに使う。
  実証: `img20260427_0001.pdf` idx=3〜8（OCR誤読「15/10/17/00/19」、実際のページ番号は4〜9の単桁）で、
  ジオメトリのみの復帰により全ページ実ページ番号が正しい位置に表示されるようになった。
- `confident()`（margin.py）を `nombre_box is not None` のみに簡略化（page_number必須をやめた）。
- 副作用としてクランプの床（output_margin_mm）を、ノンブルアンカーが効いている軸では0に緩和
  （本文がページ上端に極端に近い偶数ページ群で、マージン保証がノンブル共通位置を上書きしていた問題への対処）。

### ページ回転（`/Rotate`）の適用漏れを修正（横長ページ誤検出の真因）

`source.py: _extract_original()` は埋め込み画像のバイト列をロスレスにそのまま抜き出す方式のため、
**ページの `/Rotate` 属性を無視していた**。`img20260430_0002.pdf` のidx=82,84,130,110が「やたら横長で
regions数百個」に見えた件の真因はこれ（横長スキャン＋`/Rotate=270`で本来は縦長書籍ページとして
表示されるべきテーブル）。`page.get_rotation()` を見て `cv2.rotate()`（ロスレスな90度単位の入れ替え、
リサンプルなし）を適用。`pdfium`の回転適用版render()と向きが一致することを直接比較で確認。

保険として `pipeline.py: analyze_document()` に**書籍全体のドミナント向き（縦/横）と異なるページを
強制的に向きを揃える**安全網も追加（`Flag.PAGE_REORIENTED`でレビュー対象化）。実データでは
上記`/Rotate`修正だけで解消し、安全網は発火しなかった（発火0件を確認）。

### 白紙ページの誤フラグ抑制

`blank=True`（regions空 かつ 実インクなしと確認済み）のページに `MARGIN_NOT_FOUND` / `NO_TEXT` を
付与しないよう変更（`pipeline.py`）。両フラグとも「regionsが空」という同じ事実を繰り返すだけで、
白紙と確定している以上、人間が確認すべき新情報ではない。`blank=False`（regions空だが実インクあり、
OCR見逃しの疑い）のページは引き続きフラグが立つ。

### OCR再実行なしの高速イテレーションパス

`pipeline.py: compute_margin()`（共通ヘルパー化）+ `recompute_shadows_and_margins()` を追加。
regionsはOCR結果なので影/margin/コンテンツ枠ロジックの変更には依存しない → 保存済みregionsを
再利用すれば、176ページの再計算がOCR込み15-30分→OCRなし**約30秒〜2分**に短縮。
margin.py側のロジック調整は今後これで高速に検証できる。

### NPU並列化: 学んだこと（重要、再発防止）

- **Intel NPU（OpenVINOExecutionProvider, device_type="NPU"）に複数スレッドから同時に推論リクエストを
  投げると、NPUドライバそのものがクラッシュする**（`ZE_RESULT_ERROR_DEVICE_LOST`,
  "device hung, reset, was removed"）。一度発生すると、同一プロセス内ではその後の全呼び出しが
  失敗し続ける（が、`content.py`の `except Exception: ... Flag.OCR_FAILED` により**データ破損は無く、
  安全にフラグが立つだけ**だったことを確認済み）。新しいプロセスを起動すれば復旧する（永続故障ではない）。
  → これは sibling project `NDL-OCR-Lite-NPU` の `docs/reports/11_NPU_STABILITY_QUEUE_CONTROL_REPORT.md`
  / `12_QUEUE_CONTROLLED_BENCHMARK.md` で**既に報告・対策済みの既知問題**と完全に一致（彼らの対策は
  `--rec-workers 1`）。
- **対策**: `pipeline.py: analyze_document(max_workers=...)` で、NPU系device（`npu`/`qnn`/
  `openvino-auto`）は常に**ページレベルのスレッドプールを使わない**（`max_workers`指定を無視して
  強制的に1並列）。CPU/CUDA等は通常通り `ThreadPoolExecutor` で並列化（実測 ~1.7倍, 4 workers）。
- **NPU向けの安全な高速化**: `_analyze_pages_npu_pipelined()` を追加。OCR呼び出し自体は常に1つしか
  並行させないが、**次ページのデスキュー（CPU専用処理、約0.24秒）を背後スレッドで先読み**し、
  現ページのOCR呼び出し（NPU、約3.3秒）の間にNPUを待たせない。A/B実測: 直列82.16秒→
  パイプライン77.70秒（**約5.7%改善**、20ページ、warm engine）。理論値（0.24/3.81≈6.3%）とほぼ一致。
- **NPU構築コスト**: `HybridOCR(device="npu", openvino_cache_dir=...)` のモデルロード+コンパイルは
  約17〜140秒（キャッシュの温まり具合で変動、`openvino_cache_dir`指定で概ね2倍程度短縮）。
  **プロセス内グローバルキャッシュ**（`content.py: _get_ocr_engine`）により、同一プロセス内で複数冊を
  処理する場合は初回のみこのコストを払う（実証: 5冊×20ページのテストで2冊目以降は構築コストゼロ）。
- **1ページの処理時間の内訳**（`img20260430_0002.pdf`, 20ページ平均, 構築コスト除く, NPU）:
  デスキュー 0.242s / OCR(検出+認識) 3.289s(**全体の86%**) / 影除去+内容枠 0.149s / レンダリング
  0.128s。OCR以外を完全にパイプライン化しても理論上 14%程度の改善が上限（認識が検出の2.3倍重い）。
  NPU使用率がタスクマネージャー上70%程度（idle 30%）だった残りのギャップは、`hybrid_ocr`ライブラリ
  内部（認識の前処理/後処理等、別プロジェクト）のCPU処理に起因する可能性が高く、今回は未着手。

### 検証ツール（再利用可）
- `C:\tmp\improvements_review\verify_content_box.py`: 任意のDB/ページ番号を指定すると、元画像に
  content box・regions・OCRテキストを重ねたPNGを生成。「このページが大きい/おかしい理由」を
  人間が目視確認できる常設の検証手段として作成（ユーザー要望: 「人間が確認できるような検証体制」）。
- `C:\tmp\improvements_review\overlay_content_box_idx0_10.py`: 指定idx範囲のみ、OCRテキスト内容
  （Meiryoフォントで日本語描画）付きのオーバーレイPDFを生成。

### 既知の未着手・次セッションへの注意
- 今回の修正は主に `img20260427_0001.pdf` で詳細検証し、`img20260423_0001.pdf` / `img20260430_0002.pdf`
  は一部ページのみ抜粋確認（フル5冊の再処理・レビューキュー確認はまだ）。**次の一手は5冊フル再処理**
  （高速パス`recompute_shadows_and_margins`があるので、既存DBがあればOCR再実行は不要）。
- NPU並列前処理パイプラインによる5.7%改善は実装済みだが、`hybrid_ocr`内部のCPU処理（認識前処理等）
  に踏み込んだ追加最適化はユーザーと相談の上で見送り中（別プロジェクトへの変更が必要なため）。
- CIが見落としていたギャップ: `tests/test_profile_store.py::test_run_populates_profile` に
  `pytest.importorskip("hybrid_ocr")` が無く、CI（hybrid-ocr未インストール環境）で必ず失敗していた
  （`param-profile`ブランチが一度もpush/PRされていなかったため発覚しなかった）。修正済み(`ea892ef`)。
  **`pipeline.run()`を経由するテストを新規追加する際は、必ず`pytest.importorskip("hybrid_ocr")`を
  忘れないこと**（test_mrc.py/test_render_vecfill.pyと同じパターン）。

## 12. 2026-06-24 追記（マージン配置の再設計＋並列化）

### remargin 高速イテレーションパスの CLI 化＋並列化
- `hokusai-press remargin <source...> --out <dir> --db <db> [--workers N] [--pages "0-29"]`：
  保存済み OCR regions を再利用し、影/margin/content枠/ノンブルを再計算→PDF再生成。**OCRは走らない**。
  margin.py のロジック調整を 5冊フルで OCR なしに検証できる（`recompute_shadows_and_margins`＋`rebuild`）。
- `--pages` は**出力する頁のみ**を絞る（margin統計は全頁を使う＝部分出力でも判型は本全体で代表的）。
  各冊30頁程度のサブセットで素早く目視検証する運用（メモリ [[feedback-hokusaipress-verification]]）。
- **頁内並列化**：`MrcPageBuilder.add_page` を状態を持たない純粋関数 `encode_page`（G4/JPEG符号化、
  build_pdf 時間の ~65%）に分割。`build_pdf(max_workers=N)` で render+encode をスレッドプール化。
  実測 176頁で **2.78倍**（3分9秒→1分8秒、14コア）、出力バイト一致。`--process-pool` も用意したが
  フルパイプラインでは利得ほぼ無し（margin再計算側が直列のまま相殺）→ 既定はスレッド。
  GIL律速で14スレッド超は悪化（28で1分19秒）。コア数程度が最適点。
- C++/Rust 全面移植は非推奨：重い処理は既に cv2/libtiff/libjpeg/ONNX のネイティブ実装＝SIMD化済み。
  GIL を外せば 2.78→3.5-4.5倍見込めるが、投資対効果が薄い（結論済み）。

### マージン配置アルゴリズムの再設計（左右バランス重視）
実データ5冊で「本文が左に寄る」「左右余白が非対称」「ノンブル高さが左右でずれる」を順に修正。
- **本文ボックス抽出** `body_text_box()`：テキスト領域の x連続スパンから、小口側の柱（running head）・
  ノンブル等の小さな別スパンを自動除外（横書き本文＋右柱の 0430 で実証）。支配的スパンが無い縦書き
  （0427/0525）は content 全体にフォールバック＝柱を持たない本も統一的に扱える。
- **判型**：判型幅 = 本文幅 + 2×side_margin（side_margin = 本文左右マージンの**平均**の頑健統計）。
  以前は content幅（柱込み）に余白を二重加算して判型が約5%膨張＋中央化が崩れていた→ crop_size を
  本文基準（target = body + margins、content はクリップ防止の床のみ）に修正。
- **配置**：ノンブルのある側＝小口（印刷ノンブルは小口隅に置く慣行＝信頼できる側判定。左右gap大小比較は
  検出ノイズで誤判定したため廃止）。**ノンブルの外端を全頁同一の可視マージンにアンカー**（ユーザー選択＝
  「ノンブル固定優先」）。本文の小口側マージンは side_margin に揃い、本文幅の頁差は綴じ側（背に隠れ
  目立たない）が吸収。判型外にはみ出す要素はクランプで中心を最小限ずらす（判型は広げない）。
- ノンブルなし頁＝元配置比率を保存（baseline）／フォールバック＝上下左右中央。

### ノンブル左右高さズレの修正（nombre.py のスナップ廃止）
- 原因：`nombre.resolve` の `_aligned_box` がノンブル箱の中心を共通バンドにスナップしていたが、**箱を
  動かしても実際の墨は動かない**。この本は recto/verso でノンブル実位置が約18px(≈文字1個)異なり、
  スナップ箱でアンカーすると実墨がその差ぶんズレて出力されていた。
- 修正：スナップを廃止し**真の検出箱**を保存（`_aligned_box`/`align_y`/`common_my`/`POS_ALIGN_Y_THRESH`
  削除）。margin.py のアンカーが各頁の真のノンブル位置を共通の出力高さに合わせ、実墨が揃う。
  実測：ノンブル下端の左右差 0427:18→0px、0430:2px、0525:2px（@300dpi）。

### ノンブル下線の消失修正（render.py の margin fill）
- 原因：ノンブルの印刷下線（罫線）は薄いグレーで**ノンブルOCR箱の数px下＝content枠の外**にあり、
  margin fill（content枠外を白化）が消していた。**旧版でも同様＝既存仕様**（今セッションの退行ではない）。
- 修正：margin fill の塗り境界を「content ∪（ノンブル箱を自身の高さ60%分拡張）」に。下線/上線/はみ出しを保護。
  実機で 0525 の下線復活を確認。

### 既知の未解決（次セッションへ）
- **ソースにノンブルが無い/極薄で未検出の頁**（実例：0430 idx22 は面積9px級の極薄スペックのみ、
  0525 idx20 は完全空白）。旧版でも同じ＝退行ではない。**章扉等で元から番号が無い頁**か検出漏れか不明。
  → (a) 許容 / (b) レビューフラグ立て、をユーザーと要相談。
- **0430 のノンブルが本来位置より高い**：垂直アンカーが「**本文ボックスの下マージン(107px)**」で
  ノンブルを置いており、ノンブル本来の下マージン(31px)より高く配置（測定で確認）。寸法上の不整合だが
  本文位置も連動するため未修正。ノンブルの自前下ギャップで再アンカーするかは要判断。
- **5冊フル再処理・本番DB(final2.db)反映は未実施**。検証は各冊30頁サブセットの目視まで（結果は
  `C:\tmp\tmp0613\verify30c_out\` に保持）。これらの修正を全頁で確認後に本番反映する。
