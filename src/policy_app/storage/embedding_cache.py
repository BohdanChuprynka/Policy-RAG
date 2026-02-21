from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import hashlib
import sqlite3
import time

import numpy as np

from policy_app.config import settings


def make_cache_key(text: str) -> str:
    payload = f"{settings.embed_model}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_many(keys: List[str]) -> Dict[str, np.ndarray]:
    if not settings.embedding_cache_enabled or not keys:
        return {}

    placeholders = ",".join(["?"] * len(keys))
    query = f"SELECT key, dim, vector FROM embedding_cache WHERE key IN ({placeholders})"

    out: Dict[str, np.ndarray] = {}
    with _connect() as conn:
        for key, dim, vector in conn.execute(query, keys):
            arr = np.frombuffer(vector, dtype=np.float32)
            if arr.shape[0] != dim or dim != settings.embed_dim:
                continue
            out[key] = arr.copy()
    return out


def put_many(vectors_by_key: Dict[str, np.ndarray]) -> None:
    if not settings.embedding_cache_enabled or not vectors_by_key:
        return

    now = int(time.time())
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO embedding_cache(key, model, dim, vector, created_unix)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (k, settings.embed_model, int(v.shape[0]), v.astype(np.float32).tobytes(), now)
                for k, v in vectors_by_key.items()
            ],
        )
        conn.commit()


def _db_path() -> Path:
    if settings.embedding_cache_path:
        return Path(settings.embedding_cache_path)
    return settings.data_dir / ".cache" / "embeddings.sqlite3"


def _connect() -> sqlite3.Connection:
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_cache (
            key TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL,
            created_unix INTEGER NOT NULL
        )
        """
    )
    return conn

