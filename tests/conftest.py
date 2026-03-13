import os

# Keep tests independent from developer-local .env files.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import pytest
import fakeredis

import policy_app.storage.session_store as session_store


@pytest.fixture(autouse=True)
def patch_redis(monkeypatch):
    """Swap the real Redis pool with an in-memory fake so tests
    run without a Redis server."""
    fake = fakeredis.FakeAsyncRedis()
    monkeypatch.setattr(session_store, "_pool", fake)
    yield
