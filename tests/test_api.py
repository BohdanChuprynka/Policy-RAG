from fastapi.testclient import TestClient

from policy_app.api import app, STATE


def test_health_starts_empty():
    STATE["pipeline"] = None 
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["has_pipeline"] is False
    assert data["num_chunks"] == 0


def test_query_requires_ingest():
    STATE["pipeline"] = None
    client = TestClient(app)

    r = client.post("/query", params={"question": "test"})
    assert r.status_code == 400
    assert "No documents ingested yet" in r.json()["detail"]


def test_ingest_pdf_then_query(tmp_path):
    # NOTE: you need a small PDF fixture in your repo, or generate one.
    STATE["pipeline"] = None
    client = TestClient(app)

    pdf_path = "tests/fixtures/McDonalds_Policy.pdf" # change if needed
    with open(pdf_path, "rb") as f:
        r = client.post("/ingest/pdf", files={"file": ("sample.pdf", f, "application/pdf")})

    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["added_chunks"] >= 1
    assert data["total_chunks"] >= 1

    q = client.post("/query", params={"question": "What is this document about?"})
    assert q.status_code == 200
    out = q.json()
    assert "answer" in out
    assert "sources" in out