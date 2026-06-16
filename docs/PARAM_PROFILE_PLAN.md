# パラメータ・プロファイル計画（自己校正 × 書誌メタデータ）

状態: 計画確定（2026-06-16）。実装は Phase 0 から。
目的: 「1ページで決めない」「冊子の統計でパラメータを育てる」「スキャナ/紙/ジャンルに
馴染ませる」「人手補正を教師信号に」を、**非破壊パラメータ駆動**の上に実装する。

## 0. 原則
- **測定は各頁／決定は冊子**。各頁は raw を測るだけ。しきい値・正規化・外れ値判定・受理条件は
  冊子（複数画像）の統計で決める。
- **3層パラメータ**: prior（バケツ・永続・共有可）→ posterior（この冊子・実行ごと推定）→ page（測定値）。
- **外部メタデータ＝スコープキー**: ジャンル・判型・著者・年代で「どのバケツ（事前分布）か」を選び、
  スコープ汚染（縦書き小説と横書き教科書の混在等）を断つ。
- **前後数ページを読む**目的は2つ: (a) 識別子抽出（奥付・ISBN・タイトル）、(b) 構造分割
  （cover/前付け/本文/後付け）。読む枚数は**決め打ちせず逐次判定**。
- **非破壊**だから posterior を後から再推定→無劣化再レンダで「育てて再生成」が安全。

## 1. スコープキー＝階層バックオフ（欠損許容）
キーは固定の組でなく格子。埋まった次元だけで一致し、薄ければ親へ後退（階層ベイズの縮約）。
| レベル | キー例 | 使用条件 |
|---|---|---|
| L3 | scanner × 判型 × ジャンル | 標本 n≥閾値 |
| L2 | scanner×判型 / scanner×ジャンル | L3 が薄い |
| L1 | scanner / 判型 | さらに薄い |
| L0 | グローバル prior | cold start |
- どの次元も任意。ジャンル未取得でも L2/L1 で機能（欠損＝後退）。

## 2. 保存項目（著作権セーフ＝本文テキストは保存しない）
幾何・構造・統計のみ保存。学習に必要な情報と保存可能な情報が一致し、将来のネット共有も安全。
| 区分 | 保存する | 保存しない |
|---|---|---|
| ページ | index, is_ocr, region_count, deskew(角/信頼), content_box, margin, ink閾値, blank, **ノンブル値+枠** | 本文テキスト |
| 領域 | class(text/figure/photo), 枠(正規化), conf, source | **ocr_text(本文)** |
| 冊子(posterior) | 判型分布, ノンブル位置モデル(奇偶x/y), 支配offset, front/body/back境界, クラス密度, 総頁/OCR頁 | — |
| バケツ(prior) | 上記の robust 統計(median/MAD/n) | — |
- ノンブル「値」は頁番号という事実情報なので保存可。本文の文章は不可。

## 3. 保存粒度とOCR最小化（現段階 = (b) 軽量・冊数優先）
**「全頁“保存”」と「全頁“OCR”」を分離する**。page_feature の行保存は激安、高いのはOCR。
page_feature を2ティアに分ける:
| ティア | 項目 | OCR | 範囲 |
|---|---|---|---|
| 幾何 | page size(判型), deskew角/信頼, content枠(インク基準) | 不要(raster) | **全頁**（`--no-ocr`経路で安価） |
| OCR | region数/クラス, nombre値/枠, is_ocr, blank | 必要 | **必要頁のみ**（非OCR頁は is_ocr=False, OCR項目=None で行だけ） |

- `region_feature` は**代表頁のみ**（フラグ頁＋構造境界頁＋少数サンプル）。
- **OCRは②の停止則で最小化**: front/back を本文レジーム検出まで＋本文を一定間隔でサンプル、
  ノンブル位置モデルの分散が閾値以下＆アンカー十分で停止。非OCR頁の番号/位置はモデルで**推定**（保存しない/推定フラグ）。
- **相乗効果**: バケツ(prior)が育つほど新刊は少ないOCR頁で確信に到達 → 「育てる」＝「OCRを減らす」。
- 2モード: **校正のみ**=最小OCRサブセット／**製品(検索可能PDF)**=テキスト層のため全頁OCR→OCR項目も副産物で全頁。
- 多数の冊子でバケツ統計を太らせる breadth を優先。後で (a) 全頁領域保存へ昇格可能。

## 4. データモデル（既存 store.py SQLite を拡張）
| テーブル | 主な列 |
|---|---|
| `book` | book_id, isbn, title, author, publisher, year, ndc/ジャンル, 判型, source, confidence, resolved_at |
| `scan_profile` | book_id, scanner_sig, front/body/back境界, page_count, ocr_pages, **param_snapshot(再現性)**, created_at |
| `page_feature` | book_id, page_index, is_ocr, region_count, deskew_angle, deskew_conf, content_box, margin, ink_thr, nombre_value, nombre_box, blank |
| `region_feature` | book_id, page_index, class, box(正規化), conf, source（**text列なし**, 代表頁のみ） |
| `bucket_profile` | scope次元…, param_name, median, mad, n, updated_at |
| `decision` | 既存（人手補正＝教師信号） |

## 5. 処理フロー
冊子入力 → ①構造分割(front/body/back) → ②識別子抽出(奥付/ISBN/タイトル) →
③書誌解決(OpenBD→NDL→Google Books, 取れねば奥付OCR) → ④パラメータ解決器
(scope key で prior を引く＋本文頁の2パス統計で posterior) → ⑤既存パイプライン(無劣化レンダ) →
⑥レビュー(人手補正→教師信号) → posterior/補正をバケツへ保守的に還元(EMA＋ガード)。

## 6. 段階計画
| Phase | 内容 | web |
|---|---|---|
| 0 | `profile.py`(純ロジック: 特徴抽出/robust集約/階層バックオフ) ＋ store拡張(上記テーブル)。最初の住人=判型・ノンブル位置・ink閾値 | 不要 |
| 1 | 構造分割(front/body/back)。本文統計に限定（ローマ→アラビア遷移等の既存信号を活用） | 不要 |
| 2 | 識別子抽出（奥付/ISBN/タイトルOCR） | 不要 |
| 3 | 書誌解決＋バケツ化（OpenBD/NDL, 確信度ゲート＋人手確認） | 要 |
| 4 | レビュー補正の還元＋保守的 prior 更新(EMA＋ガード) | 不要 |
| 5(将来) | ネット共有/連合改善 | 要 |

## 7. リスクとガード
- **レジーム別統計**（奇偶・前後・章で分布が割れる→単峰前提にしない）。
- **スコープ分離**（バケツキーで別レジームを混ぜない）。
- **信用ガード&fallback**（`too_broad`/`min_anchors` を全パラメータへ。信用できない posterior は使わず prior）。
- **再現性スナップショット**（実行時に profile を固定保存）。
- **人手補正の優先**（`decided_by`、自己校正が上書きしない）。
- **書誌誤マッチ**（ISBN取り違え→確信度ゲート＋人手確認）、**プライバシー**（識別子の外部送信）、
  **API レート/オフライン代替**。

## 8. 実装分担
重い生成は Gemini/Codex に委譲し、Claude が spec・統合判断・テスト検証を担う。
delicate な store.py/pipeline 統合は Claude が直接。純ロジック(profile.py)は委譲＋検証。
