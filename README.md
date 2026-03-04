# Policy RAG App
[![CI](https://github.com/BohdanChuprynka/Policy-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/BohdanChuprynka/Policy-RAG/actions/workflows/ci.yml)

Hybrid Retrieval-Augmented Generation (RAG) assistant for policy documents.

This project ingests policy content (seed TXT and optional PDFs), builds dense + lexical indexes, retrieves grounded evidence, and generates citation-aware answers through a FastAPI backend with Streamlit frontends.

## Purpose
New hires and frontline teams lose time when policy answers are buried in long documents, especially during uncommon situations.  
This project turns company policy into a practical decision assistant: employees can ask what to do in a specific case and get a grounded answer with citations to the source policy text.

Why this is valuable:
- Improves speed and confidence for less experienced employees.
- Reduces inconsistent policy interpretation across teams.
- Supports safer, more compliant day-to-day decisions with traceable evidence.

## Table of Contents
- [Recruiter Pitch](#recruiter-pitch)
- [Overview](#overview)
- [Frontend Screenshots](#frontend-screenshots)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Run Modes](#run-modes)
- [Docker](#docker)
- [Advanced Usage](#advanced-usage)
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

## Frontend Screenshots
Main page:

![Policy RAG main page](docs/screenshots/ui-main_page.png)

Answer/output view:

![Policy RAG output](docs/screenshots/ui-output.png)

## System Architecture
C4 architecture guide: [`docs/architecture/c4-rag-architecture.md`](docs/architecture/c4-rag-architecture.md)

| C4 Level | Scope | Status | Artifact |
|---|---|---|---|
| L1 | System Context | Available | [`docs/architecture/l1-policy-architecture.png`](docs/architecture/l1-policy-architecture.png) |
| L2 | Container | Available | [`docs/architecture/l2-policy-architecture.png`](docs/architecture/l2-policy-architecture.png) |
| L3 | Component (FastAPI backend) | Available | [`docs/architecture/l3-policy-architecture.png`](docs/architecture/l3-policy-architecture.png) |

All architecture assets are stored in `docs/architecture/`.

### C4 Diagram Previews
L1 - System Context

![C4 L1 - System Context](docs/architecture/l1-policy-architecture.png)

L2 - Container

![C4 L2 - Container](docs/architecture/l2-policy-architecture.png)

L3 - Component (FastAPI Backend)

![C4 L3 - Component](docs/architecture/l3-policy-architecture.png)

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
git clone <https://github.com/BohdanChuprynka/Policy-RAG.git>
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
REPO_URL="https://github.com/BohdanChuprynka"
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

## Docker
The repository includes a production-leaning multi-stage `Dockerfile` and a `docker-compose.yml` service for the FastAPI backend.

### Build and run with Docker Compose
```bash
cp .env.example .env
# edit OPENAI_API_KEY in .env
docker compose up --build
```

Backend will be available at:
- `http://127.0.0.1:8000/health`

### Run Streamlit with Dockerized backend
Run Streamlit locally in a second terminal:
```bash
streamlit run src/streamlit_app.py
```

### Notes for Docker workflows
- Compose currently starts the backend service only.
- Backend container port mapping: `8000:8000`.
- Runtime uses non-root user `app` (UID 10001) in the container.
- If you change dependencies (`pyproject.toml` / `uv.lock`), rebuild the image with `--build`.

## Advanced Usage
Minimal advanced knobs you can tune beyond the API key.

### 1) `.env` customization
Common variables:

```dotenv
# Required
OPENAI_API_KEY="your-key"

# Network/UI
API_BASE_URL="http://127.0.0.1:8000"
REPO_URL="https://github.com/BohdanChuprynka"

# Data and ingest
SEED_POLICY_TXT="data/seed_policy.txt"
ALLOW_PDF_INGEST="false"
MAX_UPLOADS_PER_SESSION="2"
MAX_UPLOAD_MB="15"

# Retrieval/generation quality/cost
ALPHA="0.5"
HYBRID_TOPK="5"
MAX_TOP_K="10"
MODEL_NAME="gpt-4o-mini"
EMBED_MODEL="text-embedding-3-large"
MAX_TOKENS="700"

# Chunking behavior
CHUNK_SIZE="800"
CHUNK_OVERLAP="100"
MIN_CHUNK_CHARS="100"

# Session and request limits
SESSION_TTL_SECONDS="3600"
MAX_QUESTION_CHARS="600"
REQUEST_TIMEOUT_SHORT="15"
REQUEST_TIMEOUT_LONG="60"

# Embedding cache
EMBEDDING_CACHE_ENABLED="true"
# EMBEDDING_CACHE_PATH="data/.cache/embeddings.sqlite3"
```

### 2) Runtime query tuning (without restart)
- In Streamlit, tune `Alpha` and `Top K` from the sidebar.
- In API calls, pass `top_k` and `alpha` query params to `/query`.
- Guardrails are enforced server-side (`MAX_TOP_K`, `MAX_QUESTION_CHARS`).

### 3) Safe deployment defaults
- Keep `ALLOW_PDF_INGEST=false` for public deployments.
- Use session isolation via `X-Session-ID` when testing multi-user behavior.
- Keep embedding cache enabled to reduce repeated embedding cost/latency.

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

Run the same local quality gate used in CI:
```bash
OPENAI_API_KEY=test-key pytest -q
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
