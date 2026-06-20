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
