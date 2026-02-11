"""
app/api.py

Goal:
- HTTP layer only (transport).
- All core logic lives in pipelines/ (data_pipeline + rag_pipeline).
Latest practice:
- Use FastAPI lifespan for startup/shutdown (app.on_event is deprecated-ish in practice).


Endpoints (MVP):
- GET  /health
- POST /ingest/pdf
- POST /query
"""

from __future__ import annotations

from typing import Optional, Dict
from contextlib import asynccontextmanager
from pathlib import Path
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException

from policy_app.models import QueryResult, PipelineData
from policy_app.config import ALPHA, HYBRID_TOPK
from pipelines.data_pipeline import data_pipeline, extend_pipeline  
from pipelines.rag_pipeline import rag_pipeline


# MVP in-memory state (later: persistent storage + org scoping) # TODO: change to production level later
STATE: Dict[str, object] = {
    "pipeline": None,  # Optional[PipelineData]
}


def _require_pipeline() -> PipelineData:
    """
    Helper:
    - return STATE["pipeline"] if present
    - else raise HTTP 400
    """
    p = STATE["pipeline"]

    if p is None:
        raise HTTPException(status_code=400, detail="No documents ingested yet.")

    return p


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_env = os.getenv("SEED_POLICY_TXT")

    if seed_env:
        seed_path = Path(seed_env)
  
        if seed_path.is_file():
            STATE["pipeline"] = data_pipeline(seed_txt=seed_path, pdf_path=None)    
    yield

app = FastAPI(title="Policy RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    p = STATE["pipeline"]

    return {
      "ok": True,
      "has_pipeline": p is not None,
      "num_chunks": len(p.chunks) if p else 0
    }


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF uploads are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        chunks_pipeline = data_pipeline(seed_txt=None, pdf_path=tmp_path)

        if STATE["pipeline"] is None:
            STATE["pipeline"] = chunks_pipeline
        else:
            STATE["pipeline"] = extend_pipeline(STATE["pipeline"], chunks_pipeline.chunks)

        added = len(chunks_pipeline.chunks)
        total = len(STATE["pipeline"].chunks)

        return {"ok": True, "added_chunks": added, "total_chunks": total}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.post("/query", response_model=QueryResult)
def query(question: str, top_k: int = HYBRID_TOPK, alpha: float = ALPHA) -> "QueryResult":
    p = _require_pipeline()
    return rag_pipeline(p, question, top_k=top_k, alpha=alpha)