# WP-71 (Codex): ローカルユーザーアカウント（store 層）

家庭内サーバのマルチユーザー化・第1段。担当: Codex。認証の HTTP 配線は別 WP
（WP-73）で行う。今回は **store.py にユーザー管理のプリミティブを追加するだけ**。

## ゴール
受け入れテスト `tests/test_users.py` を **すべて green** にする。
`src/hokusai_press/store.py` を拡張する。

## 契約（逸脱不可）
1. `_SCHEMA` に `user` テーブルを追加:
   `CREATE TABLE IF NOT EXISTS user (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, salt TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user', created_at REAL NOT NULL)`
2. パスワードハッシュは **標準ライブラリのみ**（新規依存を増やさない）。
   `hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200_000)` を使い、
   `salt = secrets.token_hex(16)` をユーザーごとに生成して一緒に保存する。
3. `Store.create_user(username: str, password: str, role: str = "user") -> None`:
   - 既に同名ユーザーがいれば例外を送出（`sqlite3.IntegrityError` で良い。
     PRIMARY KEY 制約に任せる）。
4. `Store.get_user(username: str) -> dict | None`: `user` テーブルの行を
   `dict(row)` で返す（無ければ `None`）。パスワードは返してよいが
   **平文は絶対にどこにも保存・返却しない**（ハッシュのみ）。
5. `Store.verify_user(username: str, password: str) -> dict | None`:
   - ユーザーが存在し、入力パスワードのハッシュが一致すれば `get_user` と
     同じ形の dict を返す。存在しない/不一致なら `None`。
   - `hmac.compare_digest` で定数時間比較する（タイミング攻撃対策）。
6. `Store.list_users() -> list[dict]`: 全ユーザーを `username` 昇順で返す
   （`dict(row)`、パスワードハッシュ列を含めてよい。フィルタは呼び出し側の責務）。

## 模倣する既存の流儀
- 他の `save_*`/`get_*` メソッドと同じく `with self._lock:` で DB アクセスを囲む。
- スキーマは `_SCHEMA` 文字列の末尾に追記。

## 触ってよいファイル
- `src/hokusai_press/store.py`
- `docs/HANDOFF.md`（DoD 追記のみ）

## 触ってはならないファイル（契約）
- `tests/test_users.py`, その他すべて

## 検証（完了条件）
```
python -m pytest tests/test_users.py -q     # 7 passed
python -m pytest -q                          # 回帰なし
```

## DoD
- 上記2コマンド green（出力貼付）。`store.py` 以外の実装変更なし。
- `docs/HANDOFF.md` 先頭に3行追記。
