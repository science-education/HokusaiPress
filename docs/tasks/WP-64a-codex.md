# WP-64a (Codex): 回転・傾き調整 API とフラグ説明 API

担当: Codex。前提: `model.rotate_page_params` / `PageParams.rotation` /
`model.FLAG_INFO` は実装済み(Claude)。ローダの回転適用も済み。

## ゴール
受け入れテスト `tests/test_webui_adjust.py` を **すべて green** にする。
`src/hokusai_press/webui/app.py` に3エンドポイントを追加。

## 契約（逸脱不可）
1. `GET /api/flags` → `model.FLAG_INFO` をそのまま JSON で返す。
2. `POST /api/page/{doc_id}/{page_index}/rotate`（body `{"delta": int, "decided_by": str}`）:
   - ページ無し → 404。`delta` が 90/180/270 以外 → 400。
   - `_load_original(row.params)` で現在フレームの `(h, w)` を取り、
     `rotate_page_params(params, delta, w, h)` を呼ぶ（順序注意: width=w, height=h）。
   - `store.log_decision(..., field="rotation", old_value=<元rotation>,
     new_value=<新rotation>, features=page_features(params))` を記録し
     `store.upsert_page`。戻り値 `{"status": ..., "rotation": params.rotation}`。
   - 注意: `_load_original` のキャッシュキーには rotation が含まれているので、
     回転後の再描画は自動的に新しい向きになる。
3. `POST /api/page/{doc_id}/{page_index}/deskew`（body `{"angle_deg": float, "decided_by": str}`）:
   - ページ無し → 404。`abs(angle_deg) > 15` → 400（微小傾き補正のみ）。
   - `params.deskew.angle_deg = angle_deg`、`field="deskew"` で decision 記録、
     `upsert_page`。戻り値 `{"status": ..., "angle_deg": ...}`。

## 模倣する既存の流儀
- 既存の `set_content`/`decide` エンドポイントの書き方（Pydantic body、404、
  `page_features`、`log_decision`→`upsert_page`）に合わせる。

## 触ってよいファイル
- `src/hokusai_press/webui/app.py` / `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_webui_adjust.py`, `model.py`, `source.py`, `pipeline.py`, その他すべて

## 検証（完了条件）
```
python -m pytest tests/test_webui_adjust.py -q     # 6 passed
python -m pytest -q                                 # 回帰なし
```

## DoD
- 上記2コマンド green（出力貼付）。`app.py` 以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。
