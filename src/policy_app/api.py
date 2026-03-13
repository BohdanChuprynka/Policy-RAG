from __future__ import annotations

from contextlib import asynccontextmanager
import uuid
from pathlib import Path
import tempfile
import logging
import os

from fastapi import FastAPI, UploadFile, File, HTTPException, Header

from policy_app.models import QueryResult
from policy_app.config import settings
from policy_app.storage import session_store
from policy_app.pipelines.data_pipeline import data_pipeline, extend_pipeline
from policy_app.pipelines.rag_pipeline import rag_pipeline

SESSION_HEADER = "X-Session-ID"
logger = logging.getLogger(__name__)


def _resolve_session_id(session_id: str | None) -> str:
    if session_id and session_id.strip():
        sid = session_id.strip()
        if sid == "seed":
            return str(uuid.uuid4())
        return sid

    return str(uuid.uuid4())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to Redis
    await session_store.init()

    # Build and persist seed pipeline
    seed_path = settings.seed_policy_txt or (settings.data_dir / "seed_policy.txt")

    if seed_path and Path(seed_path).is_file():
        try:
            pipeline = await data_pipeline(seed_txt=str(seed_path), pdf_path=None)
        except ValueError as exc:
            logger.warning("Seed policy exists but produced no chunks: %s", exc)
        except Exception:
            logger.exception("Seed policy failed to load from %s", seed_path)
        else:
            await session_store.save_session("seed", pipeline, uploads=0)
    else:
        logger.warning("Seed policy file not found at %s", seed_path)

    yield

    # Graceful shutdown
    await session_store.close()


app = FastAPI(title="Policy RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    num_sessions = await session_store.session_count()
    has_seed = await session_store.has_seed()
    total_chunks = await session_store.seed_chunk_count() if has_seed else 0

    return {
        "ok": True,
        "num_sessions": num_sessions,
        "has_seed": has_seed,
        "total_chunks": total_chunks,
    }


@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    session_id: str | None = Header(default=None, alias=SESSION_HEADER),
):
    if not settings.allow_pdf_ingest:
        raise HTTPException(403, "PDF ingest is disabled for this deployment.")

    sid = _resolve_session_id(session_id)

    file_bytes = await file.read()
    if len(file_bytes) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "File is too large for upload.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF uploads are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        existing = await session_store.get_session(sid)

        if existing is None:
            pipeline = await data_pipeline(seed_txt=None, pdf_path=tmp_path)
            uploads = 1
            await session_store.save_session(sid, pipeline, uploads)

            added = len(pipeline.chunks)
            total = len(pipeline.chunks)
        else:
            current_pipeline, current_uploads = existing

            if current_uploads >= settings.max_uploads_per_session:
                raise HTTPException(429, "Upload limit reached for this session")

            new_pipeline = await data_pipeline(seed_txt=None, pdf_path=tmp_path)
            extended = await extend_pipeline(current_pipeline, new_pipeline)

            uploads = current_uploads + 1
            await session_store.save_session(sid, extended, uploads)

            added = len(new_pipeline.chunks)
            total = len(extended.chunks)
    finally:
        os.unlink(tmp_path)

    return {
        "ok": True,
        "session_id": sid,
        "added_chunks": added,
        "total_chunks": total,
        "uploads_used": uploads,
        "uploads_limit": settings.max_uploads_per_session,
    }


@app.post("/query", response_model=QueryResult)
async def query(
    question: str,
    top_k: int = settings.hybrid_topk,
    alpha: float = settings.alpha,
    session_id: str | None = Header(default=None, alias=SESSION_HEADER),
) -> QueryResult:
    sid = _resolve_session_id(session_id)

    # Try user session first, fall back to seed
    session = await session_store.get_session(sid)
    if session is None:
        session = await session_store.get_session("seed")

    if session is None:
        raise HTTPException(400, "No documents ingested yet.")

    pipeline, _ = session

    if len(question) > settings.max_question_chars:
        raise HTTPException(400, "Question too long.")

    top_k = min(top_k, settings.max_top_k)
    alpha = max(0.0, min(1.0, alpha))

    result = await rag_pipeline(pipeline, question, top_k=top_k, alpha=alpha)

    return result
