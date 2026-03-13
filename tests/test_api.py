import numpy as np
import pytest
from fastapi.testclient import TestClient

import policy_app.api as api_module
from policy_app.models import Chunk, DenseIndexMeta, LexicalIndex, PipelineData

app = api_module.app


def _make_fake_pipeline():
    """Return a minimal but valid PipelineData the session store can serialise."""
    chunk = Chunk(chunk_id="c1", text="test chunk", doc_name="sample.pdf", page=1)
    return PipelineData(
        chunks=[chunk],
        dense_meta=DenseIndexMeta(chunk_ids=["c1"]),
        dense_matrix=np.zeros((1, 3), dtype=np.float32),
        lexical=LexicalIndex(
            tokenized_docs=[["test", "chunk"]],
            doc_freq={"test": 1, "chunk": 1},
            avg_doc_length=2.0,
            chunk_ids=["c1"],
        ),
    )


def test_health_starts_empty():
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["has_seed"] is False
    assert data["total_chunks"] == 0


def test_query_requires_ingest():
    client = TestClient(app)

    r = client.post("/query", params={"question": "test"})
    assert r.status_code == 400
    assert "No documents ingested yet" in r.json()["detail"]


def test_ingest_pdf_then_query(monkeypatch):
    client = TestClient(app)

    async def fake_data_pipeline(seed_txt=None, pdf_path=None):
        return _make_fake_pipeline()

    async def fake_rag_pipeline(pipeline, question, *, top_k, alpha):
        return {
            "answer": "Mock answer.",
            "sources": [
                {
                    "doc_name": "sample.pdf",
                    "page": 1,
                    "snippet": "Mock snippet.",
                }
            ],
            "num_contexts": 1,
        }

    monkeypatch.setattr(api_module, "data_pipeline", fake_data_pipeline)
    monkeypatch.setattr(api_module, "rag_pipeline", fake_rag_pipeline)
    monkeypatch.setattr(api_module.settings, "allow_pdf_ingest", True)

    pdf_path = "tests/fixtures/McDonalds_Policy.pdf"  # change if needed
    with open(pdf_path, "rb") as f:
        r = client.post("/ingest/pdf", files={"file": ("sample.pdf", f, "application/pdf")})

    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["added_chunks"] >= 1
    assert data["total_chunks"] >= 1

    q = client.post(
        "/query",
        params={"question": "What is this document about?"},
        headers={"X-Session-ID": data["session_id"]},
    )
    assert q.status_code == 200
    out = q.json()
    assert "answer" in out
    assert "sources" in out
