# C4 Architecture — Policy RAG App

This document defines the production architecture represented by the C4 L1/L2/L3 diagrams in `docs/architecture/`.

## 1) Architecture Scope

The system provides policy question-answering with grounded evidence. It combines:
- document ingestion (seed TXT and optional PDF upload),
- hybrid retrieval (dense + lexical),
- citation-aware generation,
- session-scoped runtime state backed by Redis.

Primary runtime paths:
- **Query path:** user question → retrieval → grounded answer.
- **Ingestion path:** document input → chunking → index rebuild → session state update.

## 2) C4 Level 1 — System Context

### Elements
- `Policy User` (Person): asks questions, uploads policy PDFs, reviews citations.
- `Policy RAG App` (Software System): end-to-end ingestion, retrieval, and answer generation.
- `OpenAI API` (External System): embedding and chat-completion inference.

### Relationship Labels
1. `Policy User → Policy RAG App`: "submits policy questions and optional PDF uploads"
2. `Policy RAG App → OpenAI API`: "sends text for embedding and generation"
3. `OpenAI API → Policy RAG App`: "returns embeddings and generated response"
4. `Policy RAG App → Policy User`: "returns evidence-grounded answers with source citations"

### Context-Level Responsibilities
- The app owns retrieval and citation correctness.
- OpenAI API is used strictly for inference, not system state.
- Policy files are treated as source-of-truth inputs to indexing.

## 3) C4 Level 2 — Container Architecture

System boundary: `Policy RAG App`.

### Containers
- `Streamlit UI` (`src/frontend/streamlit_app.py`)
  - Browser-facing interaction layer.
  - Sends `X-Session-ID` with API requests.
  - Presents answer text and evidence sources.

- `FastAPI Backend` (`src/policy_app/api.py`)
  - API endpoints: `/health`, `/ingest/pdf`, `/query`.
  - Orchestrates session lifecycle, ingestion, and query execution.
  - Applies guardrails (`max_question_chars`, upload limits, `max_top_k`, `alpha` bounds).

- `Redis Session Store` (Redis 7)
  - Holds session-scoped `PipelineData` as JSON with base64-encoded NumPy arrays.
  - Sliding-window TTL per session; seed session has no expiry.
  - Deployed as a managed plugin on Railway or as `redis:7-alpine` via Docker Compose.

- `SQLite Embedding Cache` (`src/policy_app/storage/embedding_cache.py`)
  - Keyed by hash of `{embed_model + text}`.
  - Prevents repeated embedding requests.
  - Persisted at `data/.cache/embeddings.sqlite3` by default.

- `Local Data Files` (`data/`, temp upload files)
  - Seed policy source (`seed_policy.txt`).
  - Temporary upload staging for PDF ingestion.

- `OpenAI API` (external)
  - Embedding endpoint for dense index/query vectors.
  - Chat completion endpoint for final answer generation.

### Container Relationship Labels
1. `Policy User → Streamlit UI`: "asks policy questions and uploads PDFs in browser"
2. `Streamlit UI → FastAPI Backend`: "sends HTTP requests with `X-Session-ID`"
3. `FastAPI Backend → Redis Session Store`: "persists and retrieves session `PipelineData`"
4. `FastAPI Backend → Local Data Files`: "loads `seed_policy.txt` and temporary uploaded PDFs"
5. `FastAPI Backend → SQLite Embedding Cache`: "checks and persists chunk/query embeddings"
6. `FastAPI Backend → OpenAI API`: "requests `text-embedding-3-large` and `gpt-4o-mini` responses"

### Container Data Ownership
- Streamlit owns transient UI state only.
- FastAPI owns runtime orchestration and enforcement.
- Redis owns per-session retrieval artifacts with TTL lifecycle.
- SQLite cache owns reusable embedding vectors.

## 4) C4 Level 3 — Component Architecture (FastAPI Backend)

Container in focus: `FastAPI Backend`.

### Components and Responsibilities
- `API Layer` (`policy_app/api.py`)
  - Validates request constraints.
  - Resolves session identity.
  - Delegates to ingestion/query pipelines.

- `Data Pipeline` (`policy_app/pipelines/data_pipeline.py`)
  - Builds `PipelineData` from seed/PDF inputs.
  - Supports append behavior through `extend_pipeline`.

- `Loaders` (`policy_app/ingest/loaders.py`)
  - Reads TXT/PDF input.
  - Normalizes and cleans text.
  - Splits into chunks (`chunk_size`, `chunk_overlap`, `min_chunk_chars`).

- `Index Builder` (`policy_app/ingest/index_build.py`)
  - Dense index: embedding matrix + chunk ID metadata.
  - Lexical index: tokenized docs, doc frequency, avg doc length.

- `RAG Pipeline` (`policy_app/pipelines/rag_pipeline.py`)
  - Executes retrieval orchestration and answer generation.
  - Binds merged retrieval hits to canonical chunks.

- `Dense Retriever` (`policy_app/retrieval/dense.py`)
  - Embeds question and computes vector similarity.
  - Returns top-N semantic hits.

- `Lexical Retriever` (`policy_app/retrieval/lexical.py`)
  - Token-based BM25-style scoring.
  - Returns top-M lexical hits.

- `Hybrid Ranker` (`policy_app/retrieval/hybrid.py`)
  - Min-max normalization per retriever.
  - Weighted fusion: `fused = alpha * dense + (1 - alpha) * lexical`.

- `Embedding Service` (`policy_app/llm/embedding.py`)
  - Batch embedding requests.
  - Cache-first lookup, OpenAI fallback.
  - L2 normalization for dense retrieval math stability.

- `Answer Generator` (`policy_app/llm/generate.py`)
  - Constructs evidence packet with stable `[n]` source IDs.
  - Calls chat completion with constrained system prompt.
  - Parses citation markers and maps them to source metadata.

- `Embedding Cache` (`policy_app/storage/embedding_cache.py`)
  - SQLite read/write adapter for vector cache.
  - Ensures dimensional consistency checks.

- `Redis Session Store` (`policy_app/storage/session_store.py`)
  - Async Redis client for session CRUD.
  - JSON + base64 serialization (no pickle).
  - Sliding-window TTL via `EXPIRE` on access.

### Main Query Flow
1. `API Layer → RAG Pipeline`: "forwards question with bounded `top_k` and `alpha`"
2. `RAG Pipeline → Dense Retriever`: "runs semantic retrieval over normalized vector index"
3. `Dense Retriever → Embedding Service`: "embeds the user query text"
4. `Embedding Service → Embedding Cache`: "fetches cached vector or stores new vector"
5. `Embedding Service → OpenAI API`: "requests embedding when cache miss occurs"
6. `RAG Pipeline → Lexical Retriever`: "runs BM25-style retrieval over tokenized chunks"
7. `RAG Pipeline → Hybrid Ranker`: "min-max normalizes and fuses dense+lexical scores"
8. `RAG Pipeline → Answer Generator`: "passes top evidence chunks for grounded generation"
9. `Answer Generator → OpenAI API`: "requests chat completion with evidence-formatted prompt"
10. `Answer Generator → API Layer`: "returns answer, citations, and `num_contexts`"

### Ingestion Flow
1. `API Layer → Data Pipeline`: "starts ingestion for seed TXT or uploaded PDF"
2. `Data Pipeline → Loaders`: "loads files, normalizes text, removes boilerplate, chunks content"
3. `Data Pipeline → Index Builder`: "builds dense matrix and lexical metadata indexes"
4. `Index Builder → Embedding Service`: "embeds chunk texts for dense index creation"
5. `Embedding Service → Embedding Cache / OpenAI API`: "resolves embeddings via cache-first OpenAI fallback"
6. `Data Pipeline → API Layer`: "returns `PipelineData` (chunks, dense index, lexical index)"
7. `API Layer → Redis Session Store`: "stores or extends session-scoped pipeline state"

## 5) Interchange Contracts

### API Contracts
- `/ingest/pdf`:
  - Input: multipart PDF + optional `X-Session-ID`.
  - Output: session ID, added chunk count, total chunks, upload usage.
- `/query`:
  - Input: `question`, `top_k`, `alpha` + optional `X-Session-ID`.
  - Output (`QueryResult`): `answer`, `sources[]`, `num_contexts`.

### Pipeline Artifact Contract
`PipelineData` bundles:
- canonical `chunks`,
- `dense_matrix` + `dense_meta`,
- lexical index (`tokenized_docs`, `doc_freq`, `avg_doc_length`, `chunk_ids`).

This artifact is the unit of session retrieval state, serialized to Redis as JSON with base64-encoded NumPy arrays.

## 6) Runtime and Operational Characteristics

### Session Isolation and Lifecycle
- Session keying by `X-Session-ID`; fallback behavior supports seeded default session.
- Redis-native TTL with sliding window (`EXPIRE` refreshed on every access).
- Upload count limits prevent unbounded ingestion growth per session.
- Seed session persists without TTL.

### Performance Model
- Query latency is dominated by embedding/completion inference.
- SQLite cache reduces repeated embedding latency and cost.
- Dense and lexical retrieval run in-process over prepared indexes.
- Redis async operations avoid blocking the event loop.

### Consistency and Safety Controls
- `top_k` and `alpha` are clamped to configured bounds.
- Question length and upload size are bounded.
- PDF ingest can be disabled at deployment level.
- Prompt policy requires evidence-only answering and citation references.
- Session serialization uses JSON + base64 (no pickle) for safety and debuggability.

### Failure Behavior
- Missing/invalid input is rejected early at API layer.
- Empty ingestion result fails fast (`No chunks loaded` path).
- Cache dimension mismatches are skipped to avoid invalid vector reuse.
- If model output is empty, response falls back to deterministic not-found text.
- Redis connection failure at startup prevents the app from serving traffic.

## 7) Deployment

### Local Development
- `docker-compose.yml` orchestrates Redis + Backend + Streamlit.
- Backend and Streamlit each have dedicated Dockerfiles at the repo root.

### Railway (Cloud)
- Three services in a single Railway project:
  - **Redis** — Railway managed database plugin.
  - **Backend** — `Railway/Dockerfile.backend`, reads `$PORT` at runtime.
  - **Frontend** — `Railway/Dockerfile.frontend`, reads `$PORT` at runtime.
- `REDIS_URL` is injected as a reference variable from the Redis plugin.
- Internal networking (`*.railway.internal`) keeps Redis traffic off the public internet.
- See `Railway/SETUP.md` for step-by-step instructions.

## 8) Architectural Notes

- Session state is persisted in Redis for horizontal scalability and crash resilience.
- Embedding cache is persistent and optional, decoupled from session state.
- Retrieval strategy is intentionally hybrid to balance semantic recall and lexical precision.
- Citation mapping is deterministic because evidence packet ordering is stable.
