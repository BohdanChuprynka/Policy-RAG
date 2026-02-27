# Policy RAG App

Hybrid Retrieval-Augmented Generation (RAG) assistant for policy documents.

This project ingests policy content (seed TXT and optional PDFs), builds dense + lexical indexes, retrieves grounded evidence, and generates citation-aware answers through a FastAPI backend with Streamlit frontends.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Run Modes](#run-modes)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Operational Notes](#operational-notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

## Overview
- Hybrid retrieval: semantic vector search + lexical BM25-like scoring.
- Grounded generation: answers are generated from retrieved evidence only.
- Session-scoped ingestion: each session can have isolated uploaded document context.
- Multiple interfaces: FastAPI (`/health`, `/query`, `/ingest/pdf`), Streamlit app(s), CLI.

## System Architecture
```text
Documents (TXT / PDF)
        |
        v
Ingestion + Chunking
(loaders.py)
        |
        v
Index Build
- Dense embeddings index
- Lexical index
(index_build.py)
        |
        v
RAG Runtime
- dense_retrieve
- lexical_retrieve
- hybrid_merge
(rag_pipeline.py)
        |
        v
LLM Answer Generation
(generate.py)
        |
        v
Interfaces
- FastAPI
- Streamlit
- CLI
```

## Repository Structure
```text
src/
  policy_app/
    api.py                     # FastAPI app and session-based runtime state
    config.py                  # Settings from environment/.env
    models.py                  # Shared Pydantic models
    ingest/
      loaders.py               # TXT/PDF load + chunking
      index_build.py           # Dense + lexical index construction
    retrieval/
      dense.py                 # Embedding similarity retrieval
      lexical.py               # Lexical retrieval
      hybrid.py                # Fusion logic
    llm/
      embedding.py             # OpenAI embedding client + normalization
      generate.py              # OpenAI answer generation + citations
    storage/
      embedding_cache.py       # SQLite embedding cache
  pipelines/
    data_pipeline.py           # Build/extend pipeline artifacts
    rag_pipeline.py            # End-to-end retrieval + generation orchestration
  streamlit_app.py             # Local full demo UI (with PDF ingest flow)
  streamlit_recruiter_app.py   # Public-safe recruiter demo UI

pipeline_cli.py                # CLI entrypoint
tests/                         # Test suite
data/                          # Seed/sample documents
```

## Tech Stack
- Python 3.10+
- FastAPI + Uvicorn
- Streamlit
- OpenAI Python SDK
- NumPy
- Pydantic / pydantic-settings
- SQLite (embedding cache)

## Quick Start

### 1) Clone and enter project
```bash
git clone <your-repo-url>
cd RAG-App
```

### 2) Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
Preferred (project lockflow):
```bash
pip install -r requirements.txt
```

Alternative editable install:
```bash
pip install -e .
```

### 4) Configure environment
```bash
cp .env.example .env
```

Set required value in `.env`:
```dotenv
OPENAI_API_KEY="your-key"
```

Optional toggles:
```dotenv
API_BASE_URL="http://127.0.0.1:8000"
SEED_POLICY_TXT="data/seed_policy.txt"
ALLOW_PDF_INGEST="false"
EMBEDDING_CACHE_ENABLED="true"
# EMBEDDING_CACHE_PATH="data/.cache/embeddings.sqlite3"
```

## Configuration
Core settings live in `src/policy_app/config.py`.

### Retrieval and generation
- `EMBED_MODEL` (code default: `text-embedding-3-large`)
- `MODEL_NAME` (code default: `gpt-4o-mini`)
- `HYBRID_TOPK` / `ALPHA`
- `MAX_TOKENS`
- `SYSTEM_PROMPT`

### Security / limits
- `SESSION_TTL_SECONDS`
- `MAX_UPLOADS_PER_SESSION`
- `MAX_TOP_K`
- `MAX_QUESTION_CHARS`
- `MAX_UPLOAD_MB`
- `ALLOW_PDF_INGEST`

### Paths
- `SEED_POLICY_TXT`
- `EMBEDDING_CACHE_PATH`

## Run Modes

### FastAPI backend
```bash
uvicorn policy_app.api:app --app-dir src --reload --port 8000
```

Health check:
```bash
curl http://127.0.0.1:8000/health
```

### Streamlit app (full local demo)
```bash
streamlit run src/streamlit_app.py
```

### Streamlit recruiter app (safe/public-oriented)
```bash
streamlit run src/streamlit_recruiter_app.py
```

### CLI mode
Single question:
```bash
python src/pipeline_cli.py --seed-txt data/seed_policy.txt --question "What is this policy about?"
```

Interactive mode:
```bash
python src/pipeline_cli.py --seed-txt data/seed_policy.txt
```

## API Reference

### `GET /health`
Returns service and index/session summary.

Example response:
```json
{
  "ok": true,
  "num_sessions": 1,
  "has_seed": true,
  "total_chunks": 42
}
```

### `POST /query`
Parameters:
- `question` (required, query param)
- `top_k` (optional)
- `alpha` (optional)

Header:
- `X-Session-ID` (optional; used for session-scoped context)

Example:
```bash
curl -X POST "http://127.0.0.1:8000/query?question=What%20is%20prohibited%3F&top_k=5&alpha=0.5" \
  -H "X-Session-ID: my-session"
```

### `POST /ingest/pdf`
Uploads a PDF into session-scoped retrieval context.

Requirements:
- `ALLOW_PDF_INGEST=true`
- `multipart/form-data` file field name: `file`

Example:
```bash
curl -X POST "http://127.0.0.1:8000/ingest/pdf" \
  -H "X-Session-ID: my-session" \
  -F "file=@data/McDonalds_Policy.pdf"
```

## Testing
Run all tests:
```bash
pytest -q
```

If tests involving ingest are enabled, make sure your test environment matches API flags (for example, `ALLOW_PDF_INGEST`).

## Operational Notes
- The app is currently stateful in-memory for session pipelines (`STATE` in `api.py`).
- Embedding cache is persisted to SQLite under `data/.cache/` by default.
- Use environment-based toggles to keep public deployments safe (`ALLOW_PDF_INGEST=false`).
- Session IDs are header-driven; review access controls before exposing publicly.

## Troubleshooting
- `OPENAI_API_KEY` missing:
  - Ensure `.env` exists and contains a valid key.
- Backend unreachable in Streamlit:
  - Start API server and verify `API_BASE_URL`.
- PDF ingestion returns `403`:
  - Set `ALLOW_PDF_INGEST=true` for local testing.
- "No documents ingested yet":
  - Provide `SEED_POLICY_TXT` or ingest a PDF into the active session.

## Roadmap
- Persist session/index state in external store.
- Add authentication and stronger multi-tenant isolation.
- Harden upload path and rate limiting.
- Expand evaluation and retrieval quality benchmarking.
- Add CI gates (tests, lint, type checks) and release automation.
