# NPU 並列実行 設計計画 (NPU Parallel Execution Plan)

状態: **案A 実装済み**（並列パスをスレッド方式へ移行、NPU 直列ガード撤廃）。
実測: NPU `--workers 2` で 2 ファイル（240+160 頁）が**ハングせず完走**、
WALL 1715s（serial 相当 3029s）＝**約 1.77 倍**。最終更新 2026-06-14。

## 0. 要約

現状 `hokusai run --workers N --device npu` は **強制的にシリアル化**される
（commit `6fb3822` "guard against unreliable parallel NPU"）。本書は「NPU 並列は
本質的に不可能なのか、可能にする方法はあるのか」を検討し、可能とする実装方針を
詳細化する。

**結論: 本質的に不可能ではない。** 現状の防御は「並列化の方式が誤っている」ことへの
正しい対処であって、NPU 並列そのものの不能を意味しない。NPU を開くコンテキストを
1 つに限定し、並列は CPU 側で行う構成に変えれば実現できる。

## 1. 現状と根本原因

### 1.1 現状のコード

- `src/hokusai_press/cli.py:184-204`: ファイル単位の `multiprocessing.Pool`。
  各ワーカー**プロセス**が 1 ファイルを丸ごと処理。
- `cli.py:185-192`: `--device` が `npu`/`qnn` かつ並列のとき強制的に `workers=1`。
- `src/hokusai_press/content.py:46-62`: `_ocr_engine` は **プロセスごと**の
  module-global singleton。各プロセスが独立に `HybridOCR` を生成。
- OCR 実体は sibling `Yomitoku_NDL-OCR-Lite`:
  `src/hybrid_ocr/providers.py:82-83` が onnxruntime + `OpenVINOExecutionProvider`
  (device_type=`NPU`) で `InferenceSession` を生成。
- 検出が支配段。`docs/NPU_NOTES.md`(sibling) 実測: 固定形状 DBNet が Intel NPU で
  GO、`--device npu --openvino-cache-dir` で **CPU 比 2.4 倍 (3.01 s/page)**。
  認識のみの NPU 化は無効果（Amdahl 制約）。

### 1.2 なぜクラッシュ/ハングするのか

各ワーカープロセスが**それぞれ別の NPU コンテキストを開こうとする**ことが原因。
Intel NPU の Windows ドライバ (Level Zero / WDDM-KMD) レベルでは:

- プロセスが NPU を要求するとドライバがハードウェアコンテキスト・コマンドキュー・
  コンパイラバックエンドを確保する。
- 複数プロセスからの同時コンテキスト生成がドライバのメモリ管理の競合を誘発、
  または固定ハードウェアコンテキスト上限を枯渇させる。
- 結果、NPU スケジューラがデッドロックし、しばしば**システム再起動でしか復旧しない**
  ハング状態になる。

→ 「単一 NPU を複数 OS プロセスから同時に開く」のは**サポート外の使い方**。
commit `6fb3822` の防御は妥当だった。

### 1.3 正しい並列モデル

NPU は単一デバイス。**NPU を開くコンテキストは 1 つに限定**し、その 1 コンテキストに
OCR を集約しつつ、**CPU が重い他段（ラスタ化・deskew・マージン・MRC・レンダリング・
CPU 認識）を並行**させる。これがコミットが当初狙った
"overlap one file's CPU work with another's NPU OCR" を正しく実現する形。

## 2. 実装方針（3 案、推奨度順）

### ★ 案 A: スレッドプール + NPU singleton 共有（推奨・最小改修）

プロセス並列をやめ、**1 プロセス内のスレッドプール**にする。NPU コンテキストは 1 つ。

**動作原理**
- onnxruntime の `session.run()` も cv2/numpy も GIL を解放する。スレッド A が NPU で
  OCR 中に、スレッド B が CPU 段を実際に並行実行できる。
- 1 プロセス = 1 OpenVINO コンテキスト → デバイス競合ゼロ。
- pdfium は sibling 側で `_PDFIUM_LOCK` により直列化済み。スレッド化と整合的。

**改修ポイント**
1. `cli.py` の `parallel` 分岐: `mp.Pool` → `concurrent.futures.ThreadPoolExecutor`。
   NPU/qnn を強制シリアルにする 185-192 行の分岐を撤廃し、スレッド経路へ。
2. OCR エンジン singleton をスレッド間共有に。`content.py:_get_ocr_engine` の生成を
   `threading.Lock` で保護（二重コンパイル防止）。`session.run` 自体は ORT が
   スレッドセーフなので並行呼び出し可（デバイス側で直列化＝想定通り）。
3. SQLite: 現状の「ファイルごと temp db → 最後に merge」を維持すればロック競合なし。
   または `store.py` の既存ロックで単一 db 共有も可。
4. `--workers` の意味を「並行ファイル数（スレッド）」に再定義。NPU 経路でも有効化。
5. OpenVINO `--openvino-cache-dir` を実質必須運用に（プロセスが 1 つになるので
   分単位の再コンパイルは初回 1 回で済む）。

**期待効果と限界（正直な見積り）**
- NPU 上の推論自体は時分割されるため速くならない。利得は「他ファイルの CPU 段を
  NPU 時間の裏に隠す」分。
- 検出が NPU 支配（~3.01 s/page）。CPU 側（認識・レンダ等）をこの裏に隠せる割合だけ
  短縮 → 現実的には 1 ファイル時比で十数〜数十%。Amdahl 制約で頭打ち。
- リスク低・改修小。**まずこれを入れるべき。**

### 案 B: 推論サーバ型（最大スループット・大改修）

NPU を所有する**専用 1 プロセス**が `ov::AsyncInferQueue`
(`optimal_number_of_infer_requests`) を回し、N 個の CPU ワーカープロセスが
decode/前処理/後処理を担い、**共有メモリ (`multiprocessing.shared_memory`) で
テンソルを受け渡す**。

- 長所: 単一ドライバコンテキストを保ちつつ NPU を完全パイプライン化、CPU 段は真の
  多プロセス並列。理論上の上限が最も高い。
- 短所: onnxruntime EP から**ネイティブ OpenVINO API へ移行**が必要、IPC/シリアライズ/
  バックプレッシャ設計、エラーハンドリングが重い。本プロジェクト（単機バッチ、
  実測 "~2 workers が最適"）にはオーバースペック気味。
- 位置づけ: 案 A で不足し、かつ大量バッチを常用するなら投資する将来案。

### 案 C: 検出の AsyncInferQueue 化（中間・NPU 専有改善）

支配段の検出だけ onnxruntime をやめネイティブ OpenVINO `AsyncInferQueue` にし、
複数ページの検出を 1 プロセス内で非同期投入して NPU の DMA/compute 段をパイプライン化。
共有メモリ IPC 不要。案 B の効果の一部を低コストで得る中間策。ただし
`detector.py` の推論経路を二系統持つことになる。

## 3. 共通の必須注意点（どの案でも）

| 項目 | 対策 |
|---|---|
| NPU コンパイルが分単位 | `ov::cache_dir`（既存 `--openvino-cache-dir`）を必須運用化 |
| 複数 `InferenceSession`/NPU でメモリ断片化・劣化 | NPU セッションはプロセス内 1 つに限定 |
| Windows TDR（>2s でドライバリセット） | 1 推論を巨大バッチにしない（固定 1536 は実証済みで可） |
| FP32→FP16 暗黙ダウンキャスト | 検出 box の精度回帰を回帰テストで監視 |
| UMA pinned メモリ枯渇 | Async キュー長を `optimal_number_of_infer_requests` 以内に |

## 4. 推奨ロードマップ

1. **案 A を実装**（cli.py のスレッド化 + エンジン共有 + cache デフォルト化）。
   低リスクで「NPU 並列禁止」を解除し当初意図を実現。
2. ベンチ（NPU_NOTES 流儀: s/page と総時間）で利得測定。CPU 段の隠蔽率を確認。
3. 利得が頭打ちで、かつ大量バッチ常用なら **案 C → 案 B** へ段階的に投資。

## 5. 案 B / 案 C 効率見積り（テレメトリ改訂版 2026-06-14）

### 5.1 計測条件と観測

`--workers 2 --device npu`（Core Ultra 7 155U / 12C14T / RAM 15.4GB）で 5 ファイル
バッチを実行中のタスクマネージャ観測:

| 資源 | 値 | 含意 |
|---|---|---|
| NPU 0 (AI Boost) | 平均 ~55%、ピーク ~99% | **律速資源だがバブルあり**（飽和しきっていない） |
| CPU | ~33%、多くのコア空き | CPU は余剰 → **CPU 並列化（案 B 本来の狙い）は効かない** |
| メモリ | 14.3/15.4GB＝**93%** | ⚠️ スケーリングの最大制約。UMA で NPU もホスト RAM を消費 |
| iGPU (Intel Graphics) | ~22% | ヘテロ実行の余地 |

### 5.2 律速の再分解（当初見積りの訂正）

当初 NPU 実デバイス時間を ~1.0–1.5 s/page と低く見たが、実測から逆算すると過小だった:

- NPU 総 busy ≈ WALL(1715s, 2ファイル test) × 0.55 ≈ **943s** → **約 2.36 s/page が NPU 実時間**。
- これは OCR 3.01 s/page の大半。**OCR 段はほぼ NPU 律速**（det+rec 両方 NPU なので当然）。
- 非 OCR の CPU 段（~4.56 s/page）は余剰コアで裏に隠れている → CPU が空く理由。
- **単一 NPU の WALL フロア ≈ 943s ≈ 直列比 3.2x**。案 A は NPU を 55% しか使えていない。

バブルの主因: 各スレッド内が「NPU 検出 → CPU クロップ → NPU 認識 → CPU 後処理」と
**直列依存**し、2 スレッドでは NPU の待ち時間を埋めきれない。

### 5.3 効率見積り

| 案 | 効く理屈 | 推定 WALL | 対直列 | 対案 A | 判定 |
|---|---|---|---|---|---|
| **案 C / 非同期フィード**（AsyncInferQueue で NPU バブルを埋める） | NPU 55%→90%+ | ~1000–1100s | ~2.8–3.0x | **~1.6–1.7x** | ◎ **本命** |
| **案 B（CPU 多プロセス農場）** | CPU を増やす | ~1600–1715s | ~1.8x | **~1.0x（無効）** | ✗ CPU は律速でない |
| **workers 増（3,4）** | 並列フィード増で NPU を埋める | 案 C 相当 | — | — | △ **メモリ 93% で頭打ち** |
| **NPU+iGPU(DML) ヘテロ**（943s フロアを破る） | det=NPU / rec=iGPU 等で device 分散 | <943s | ~4–5x | ~2.4x+ | ◇ 将来本命・別設計 |

### 5.4 結論（当初ロードマップからの転換）

- **案 B（CPU 多プロセス化）は本ワークロードでは不採用**。CPU は余り、メモリが逼迫して
  いるため、フルパイプラインを複数走らせる方向は逆効果。
- **案 C（単一 NPU を途切れなく食わせる非同期パイプライン）が本命**。バブル 45% を回収して
  ~1000–1100s（直列比 3x）を狙う。1 本の NPU 供給キューに前処理済みテンソルだけ流すため
  **メモリ効率も良く**、RAM 15.4GB の本機に適合。
- メモリ削減（per-worker 画像バッファ削減・ストリーミング）を並行課題とする。
- 真の天井破りは **NPU+iGPU(DML) ヘテロ実行**。FP16 精度差検証が前提。

## 6. 出典

- 本リポジトリ: `cli.py`, `content.py`（コード実測 2026-06-14）。
- sibling `Yomitoku_NDL-OCR-Lite`: `providers.py`, `detector.py`, `source.py`,
  `docs/NPU_NOTES.md`（NPU 実機知見）。
- Intel NPU + OpenVINO + onnxruntime のドライバ/並行性メカニズム調査（2026-06-14）。
- 案 A 実測（WALL 1715s vs serial 3029s = 1.77x）＋ `--workers 2` 実行中の
  タスクマネージャ観測（NPU ~55%/CPU ~33%/RAM 93%、2026-06-14）。
