"""
Redis-backed session store.

Each session is a single Redis key (``session:<id>``) holding a JSON blob.
NumPy arrays are base64-encoded so that nothing uses pickle — the
serialisation is explicit, safe, and debuggable.

TTL is managed by Redis natively: user sessions expire after
``settings.session_ttl_seconds``; the seed session has no expiry.
On every access we call EXPIRE to reset the TTL (sliding window).
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Tuple

import numpy as np
import redis.asyncio as aioredis

from policy_app.config import settings
from policy_app.models import (
    Chunk,
    DenseIndexMeta,
    LexicalIndex,
    PipelineData,
)

logger = logging.getLogger(__name__)

_pool: aioredis.Redis | None = None


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

async def init(url: str | None = None) -> None:
    global _pool
    url = url or settings.redis_url
    _pool = aioredis.from_url(url, decode_responses=False)
    await _pool.ping()
    logger.info("Redis session store connected: %s", url)


async def close() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


def _redis() -> aioredis.Redis:
    if _pool is None:
        raise RuntimeError("Session store not initialised — call init() first")
    return _pool


# ------------------------------------------------------------------
# Keys
# ------------------------------------------------------------------

def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


# ------------------------------------------------------------------
# Serialisation helpers
# ------------------------------------------------------------------

def _serialize(pipeline: PipelineData, uploads: int) -> bytes:
    return json.dumps({
        "chunks": [c.model_dump() for c in pipeline.chunks],
        "dense_meta": pipeline.dense_meta.model_dump(),
        "dense_matrix_b64": base64.b64encode(
            pipeline.dense_matrix.astype(np.float32).tobytes()
        ).decode(),
        "dense_matrix_shape": list(pipeline.dense_matrix.shape),
        "lexical": pipeline.lexical.model_dump(),
        "uploads": uploads,
    }).encode()


def _deserialize(raw: bytes) -> Tuple[PipelineData, int]:
    data = json.loads(raw)

    dense_matrix = np.frombuffer(
        base64.b64decode(data["dense_matrix_b64"]),
        dtype=np.float32,
    ).reshape(data["dense_matrix_shape"]).copy()

    pipeline = PipelineData(
        chunks=[Chunk(**c) for c in data["chunks"]],
        dense_meta=DenseIndexMeta(**data["dense_meta"]),
        dense_matrix=dense_matrix,
        lexical=LexicalIndex(**data["lexical"]),
    )
    return pipeline, data["uploads"]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

async def get_session(session_id: str) -> Tuple[PipelineData, int] | None:
    """Return ``(PipelineData, uploads)`` or ``None`` if the key is missing."""
    raw = await _redis().get(_session_key(session_id))
    if raw is None:
        return None
    # Reset TTL on access (sliding window) — skip for seed session.
    if session_id != "seed":
        await _redis().expire(
            _session_key(session_id), settings.session_ttl_seconds,
        )
    return _deserialize(raw)


async def save_session(
    session_id: str,
    pipeline: PipelineData,
    uploads: int,
) -> None:
    """Persist session.  Seed session has no TTL; others expire automatically."""
    raw = _serialize(pipeline, uploads)
    r = _redis()
    if session_id == "seed":
        await r.set(_session_key(session_id), raw)
    else:
        await r.setex(
            _session_key(session_id),
            settings.session_ttl_seconds,
            raw,
        )


async def has_seed() -> bool:
    return bool(await _redis().exists(_session_key("seed")))


async def seed_chunk_count() -> int:
    result = await get_session("seed")
    if result is None:
        return 0
    pipeline, _ = result
    return len(pipeline.chunks)


async def session_count() -> int:
    """Approximate active session count (uses SCAN, safe for production)."""
    count = 0
    r = _redis()
    async for _ in r.scan_iter(match="session:*", count=100):
        count += 1
    return count
