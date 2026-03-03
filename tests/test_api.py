from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import policy_app.api as api_module

app = api_module.app
STATE = api_module.STATE


@pytest.fixture(autouse=True)
def reset_api_state():
    STATE["sessions"] = {}
    yield


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
    # NOTE: you need a small PDF fixture in your repo, or generate one.
    client = TestClient(app)

    def fake_data_pipeline(seed_txt=None, pdf_path=None):
        return SimpleNamespace(chunks=[{"chunk_id": "c1"}])

    def fake_rag_pipeline(pipeline, question, *, top_k, alpha):
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

    pdf_path = "tests/fixtures/McDonalds_Policy.pdf" # change if needed
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
