"""
Module lưu trữ người dùng bằng SQLite.

Bảng `users`:
    id          INTEGER PRIMARY KEY
    name        TEXT UNIQUE         – tên người dùng
    embedding   BLOB                – vector trung bình (centroid) 192 x float32, L2-norm
    n_samples   INTEGER             – số mẫu giọng đã đăng ký
    created_at  TEXT
    updated_at  TEXT

Bảng `samples` (lưu từng embedding riêng lẻ để có thể tính lại centroid):
    id, user_id, embedding BLOB, duration REAL, created_at
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    embedding   BLOB    NOT NULL,
    n_samples   INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding   BLOB    NOT NULL,
    duration    REAL,
    created_at  TEXT    NOT NULL
);
"""


def _to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _centroid(embs: Sequence[np.ndarray]) -> np.ndarray:
    c = np.mean(np.stack(embs), axis=0)
    return (c / (np.linalg.norm(c) + 1e-10)).astype(np.float32)


class SpeakerDB:
    def __init__(self, path: str | Path = config.DB_PATH):
        self.path = str(path)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ write
    def enroll(
        self,
        name: str,
        embeddings: Sequence[np.ndarray],
        durations: Optional[Sequence[float]] = None,
        append: bool = True,
    ) -> int:
        """
        Đăng ký người dùng mới hoặc bổ sung mẫu cho người đã có.

        Embedding đại diện = trung bình các mẫu rồi chuẩn hóa L2 lại.
        Trả về tổng số mẫu của người dùng sau khi đăng ký.
        """
        name = name.strip()
        if not name:
            raise ValueError("Tên người dùng không được để trống.")
        if not embeddings:
            raise ValueError("Cần ít nhất 1 mẫu giọng nói.")
        durations = list(durations or [None] * len(embeddings))
        now = _now()

        with self._conn() as c:
            row = c.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()
            if row and not append:
                c.execute("DELETE FROM samples WHERE user_id = ?", (row[0],))
            if row:
                uid = row[0]
            else:
                uid = c.execute(
                    "INSERT INTO users(name, embedding, n_samples, created_at, updated_at) "
                    "VALUES (?, ?, 0, ?, ?)",
                    (name, _to_blob(embeddings[0]), now, now),
                ).lastrowid
            c.executemany(
                "INSERT INTO samples(user_id, embedding, duration, created_at) VALUES (?, ?, ?, ?)",
                [(uid, _to_blob(e), d, now) for e, d in zip(embeddings, durations)],
            )
            all_embs = [_from_blob(b) for (b,) in c.execute(
                "SELECT embedding FROM samples WHERE user_id = ?", (uid,))]
            c.execute(
                "UPDATE users SET embedding = ?, n_samples = ?, updated_at = ? WHERE id = ?",
                (_to_blob(_centroid(all_embs)), len(all_embs), now, uid),
            )
            return len(all_embs)

    def delete(self, name: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM users WHERE name = ?", (name,)).rowcount > 0

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM samples")
            c.execute("DELETE FROM users")

    # ------------------------------------------------------------------- read
    def get(self, name: str) -> Optional[np.ndarray]:
        with self._conn() as c:
            row = c.execute("SELECT embedding FROM users WHERE name = ?", (name,)).fetchone()
        return _from_blob(row[0]) if row else None

    def names(self) -> List[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT name FROM users ORDER BY name COLLATE NOCASE")]

    def all_embeddings(self) -> Dict[str, np.ndarray]:
        with self._conn() as c:
            return {n: _from_blob(b) for n, b in c.execute("SELECT name, embedding FROM users")}

    def list_users(self) -> List[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, n_samples, created_at, updated_at FROM users "
                "ORDER BY created_at"
            ).fetchall()
        return [dict(name=r[0], n_samples=r[1], created_at=r[2], updated_at=r[3]) for r in rows]

    def __len__(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
