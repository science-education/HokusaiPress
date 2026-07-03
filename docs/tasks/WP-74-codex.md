# WP-74 (Codex): 蔵書所有権を閲覧・一覧系ルートに配線

家庭内サーバのマルチユーザー化・第4段。前提: WP-72（`Store.set_doc_owner`/
`get_doc_owner`/`doc_ids_for_user`）, WP-73（セッション認証、`_session_user(request)`
ヘルパーが `app.py` 内に存在）完了済み。担当: Codex。

## ゴール
受け入れテスト `tests/test_doc_owner_wiring.py` を **すべて green** にする。
`src/hokusai_press/webui/app.py` を変更する。**今回のスコープは閲覧・一覧系のみ**
（`/api/library`, `/api/queue`, `/api/search`, `/read/{doc_id}`,
`/page/{doc_id}/{page_index}`, `/img/...`）。編集系エンドポイント
（decide/region/content/nombre）は対象外（別 WP、`docs/HANDOFF.md` に明記済み）。

## 契約（逸脱不可）
1. **取込時の所有権割当**: `POST /api/ingest` と `POST /api/upload` は、
   ジョブ完了後（`runner` 成功時）に `_session_user(request)` が非 `None` なら
   `store.set_doc_owner(doc_id, user["username"])` を呼ぶ。未ログイン
   （`_session_user` が `None`、典型的には `require_auth=False`）なら呼ばない
   （後方互換）。
2. **一覧のフィルタ**（`_session_user(request)` が非 `None` のときだけ適用。
   `None` なら今まで通り無フィルタ）:
   - `GET /api/library`: `store.doc_ids_for_user(username)` に含まれる doc、
     **かつ所有者が誰も割り当てられていない doc**（`get_doc_owner(doc_id) is None`）
     を含める。他ユーザー所有の doc は除外。
   - `GET /api/queue`: 同様に、doc ごとの絞り込みで他ユーザー所有 doc を除外
     （所有者なし doc は含める）。
   - `GET /api/search`: `store.search(...)` の結果から、他ユーザー所有 doc の
     ヒットを除外してから返す（所有者なし doc のヒットは含める）。
3. **閲覧ルートのアクセス制御**（`_session_user(request)` が非 `None` のときだけ
   適用。未ログインなら今まで通り）:
   - `GET /read/{doc_id}`, `GET /page/{doc_id}/{page_index}`,
     `GET /img/{doc_id}/{page_index}/analysis.png`,
     `GET /img/{doc_id}/{page_index}/output.png`:
     `owner = store.get_doc_owner(doc_id)`。`owner is not None and owner !=
     user["username"]` なら `HTTPException(403)`。`owner is None`（所有者未設定）
     なら誰でも閲覧可（後方互換／レガシーデータ）。

## 模倣する既存の流儀
- `_session_user(request)` を各ルート関数に `request: Request` 引数を足して呼ぶ
  （WP-73 で実装済みのヘルパーをそのまま使う。再実装しない）。
- 既存のフィルタ・ループ構造にオーナーチェックを1行足す形で、大きな書き換えは
  避ける。

## 触ってよいファイル
- `src/hokusai_press/webui/app.py`
- `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_doc_owner_wiring.py`, `store.py`, その他すべて
- 編集系ルート（decide/region/content/nombre 等）は今回変更しない

## 検証（完了条件）
```
python -m pytest tests/test_doc_owner_wiring.py -q     # 5 passed
python -m pytest -q                                      # 回帰なし
```

## DoD
- 上記2コマンド green（出力貼付）。`app.py` 以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。編集系エンドポイントの所有権チェックが
  未実装である旨も一言明記（次の WP の種として）。
