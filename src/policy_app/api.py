from __future__ import annotations

from typing import Dict
from contextlib import asynccontextmanager
import uuid
from pathlib import Path
import time
import tempfile
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Header

from policy_app.models import QueryResult
from policy_app.config import settings
from pipelines.data_pipeline import data_pipeline, extend_pipeline  
from pipelines.rag_pipeline import rag_pipeline

SESSION_HEADER = "X-Session-ID"
logger = logging.getLogger(__name__)

STATE: Dict[str, dict] = {
    "sessions": {}  # session_id -> {pipeline, uploads, timestamps}
}

def _cleanup_expired_sessions() -> None:
    now = time.time()
    expired = []

    for sid, data in STATE["sessions"].items():
        if sid == "seed":
            continue
        if now - data["last_seen_unix"] > settings.session_ttl_seconds:
            expired.append(sid)

    for sid in expired:
        del STATE["sessions"][sid]
    
def _resolve_session_id(session_id: str | None) -> str:
    if session_id and session_id.strip():
        sid = session_id.strip()
        if sid == "seed":
            return str(uuid.uuid4())
        return sid

    return str(uuid.uuid4())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize session store
    STATE["sessions"] = {}

    seed_path = settings.seed_policy_txt or (settings.data_dir / "seed_policy.txt")

    if seed_path and Path(seed_path).is_file():
        try:
            pipeline = data_pipeline(seed_txt=str(seed_path), pdf_path=None)
        except ValueError as exc:
            logger.warning("Seed policy exists but produced no chunks: %s", exc)
        except Exception:
            logger.exception("Seed policy failed to load from %s", seed_path)
        else:
            STATE["sessions"]["seed"] = {
                "pipeline": pipeline,
                "uploads": 0,
                "created_unix": time.time(),
                "last_seen_unix": time.time(),
            }

    else:
        logger.warning("Seed policy file not found at %s", seed_path)

    yield

app = FastAPI(title="Policy RAG API", version="0.1.0", lifespan=lifespan)

@app.get("/health")
def health() -> dict:
    sessions = STATE.get("sessions", {})

    total_chunks = 0
    for s in sessions.values():
        pipeline = s.get("pipeline")
        if pipeline:
            total_chunks += len(pipeline.chunks)

    return {
        "ok": True,
        "num_sessions": len(sessions),
        "has_seed": "seed" in sessions,
        "total_chunks": total_chunks,
    }

@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    session_id : str | None = Header(default=None, alias=SESSION_HEADER),
):
    if not settings.allow_pdf_ingest:
        raise HTTPException(403, "PDF ingest is disabled for this deployment.")

    _cleanup_expired_sessions()
    sid = _resolve_session_id(session_id)

    file_bytes = await file.read()
    if len(file_bytes) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "File is too large for upload.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF uploads are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    session = STATE["sessions"].get(sid)

    if session is None:
        pipeline = data_pipeline(seed_txt=None, pdf_path=tmp_path)

        STATE["sessions"][sid] = {
            "pipeline": pipeline,
            "uploads": 1,
            "created_unix": time.time(),
            "last_seen_unix": time.time(),
        }

        added = len(pipeline.chunks)
        total = len(pipeline.chunks)

    else:
        if session["uploads"] >= settings.max_uploads_per_session:
            raise HTTPException(429, "Upload limit reached for this session")

        new_pipeline = data_pipeline(seed_txt=None, pdf_path=tmp_path)
        extended = extend_pipeline(session["pipeline"], new_pipeline.chunks)

        session["pipeline"] = extended
        session["uploads"] += 1
        session["last_seen_unix"] = time.time()

        added = len(new_pipeline.chunks)
        total = len(extended.chunks)

    return {
    "ok": True,
    "session_id": sid,
    "added_chunks": added,
    "total_chunks": total,
    "uploads_used": STATE["sessions"][sid]["uploads"],
    "uploads_limit": settings.max_uploads_per_session,
}

@app.post("/query", response_model=QueryResult)
def query(question: str,
          top_k: int = settings.hybrid_topk, 
          alpha: float = settings.alpha,
          session_id: str | None = Header(default=None, alias=SESSION_HEADER)
) -> QueryResult:
    _cleanup_expired_sessions()
    sid = _resolve_session_id(session_id)
    session = STATE["sessions"].get(sid) or STATE["sessions"].get("seed")

    if session is None:
        raise HTTPException(400, "No documents ingested yet.")

    if len(question) > settings.max_question_chars:
        raise HTTPException(400, "Question too long.")

    top_k = min(top_k, settings.max_top_k)
    alpha = max(0.0, min(1.0, alpha))

    result = rag_pipeline(session["pipeline"], question, top_k=top_k, alpha=alpha)
    
    session["last_seen_unix"] = time.time()

    return result
    
