# C4 Architecture - Policy RAG App 

## 2) C4 Level 1 - System Context

### People
- Policy User (employee/recruiter/demo user)

### Software System (Your system)
- Policy RAG App

### External Systems
- OpenAI API
- Local Policy Documents (seed TXT + uploaded PDFs)

### Relationships (draw arrows with these labels)
1. Policy User -> Policy RAG App: "asks policy questions and uploads PDFs"
2. Policy RAG App -> OpenAI API: "requests embeddings and answer generation"
3. Policy RAG App -> Local Policy Documents: "loads seed and uploaded policy content"
4. Policy RAG App -> Policy User: "returns grounded answers with citations"

---

## 3) C4 Level 2 - Container Diagram

System boundary: **Policy RAG App**

### Containers inside boundary
- Streamlit UI (`src/streamlit_app.py`)
  - Role: web UI for question asking and PDF upload
- FastAPI Backend (`src/policy_app/api.py`)
  - Role: ingest/query endpoints + session runtime
- In-Memory Session Store (inside FastAPI process)
  - Role: keeps per-session `PipelineData` (chunks + indexes)
- SQLite Embedding Cache (`src/policy_app/storage/embedding_cache.py`)
  - Role: caches embedding vectors to reduce repeat OpenAI calls
- Local Data Files (`data/`)
  - Role: seed policy text and temporary/ingested document content

### External container/system
- OpenAI API

### Key relationships
1. Policy User -> Streamlit UI: "uses app in browser"
2. Streamlit UI -> FastAPI Backend: "HTTP `/health`, `/ingest/pdf`, `/query`"
3. FastAPI Backend -> In-Memory Session Store: "creates/updates per-session pipeline artifacts"
4. FastAPI Backend -> Local Data Files: "loads seed TXT, reads uploaded PDF temp files"
5. FastAPI Backend -> SQLite Embedding Cache: "reads/writes embedding vectors"
6. FastAPI Backend -> OpenAI API: "embeddings + chat completion"
7. FastAPI Backend -> Streamlit UI: "query results with answer and sources"

---

## 4) C4 Level 3 - Component Diagram (FastAPI Backend)

Container in focus: **FastAPI Backend**

### Components
- API Layer (`policy_app/api.py`)
  - Endpoints: `/health`, `/ingest/pdf`, `/query`
  - Session lifecycle and limits
- Data Pipeline (`pipelines/data_pipeline.py`)
  - Orchestrates chunk loading + index building
- Loaders (`policy_app/ingest/loaders.py`)
  - TXT/PDF loading, cleaning, chunking
- Index Builder (`policy_app/ingest/index_build.py`)
  - Builds dense matrix + lexical index
- RAG Pipeline (`pipelines/rag_pipeline.py`)
  - Orchestrates retrieval and generation for a question
- Dense Retriever (`policy_app/retrieval/dense.py`)
  - Embedding similarity search
- Lexical Retriever (`policy_app/retrieval/lexical.py`)
  - BM25-like keyword scoring
- Hybrid Ranker (`policy_app/retrieval/hybrid.py`)
  - Normalizes and fuses dense + lexical scores
- Embedding Service (`policy_app/llm/embedding.py`)
  - Calls OpenAI embeddings, uses cache
- Answer Generator (`policy_app/llm/generate.py`)
  - Builds evidence prompt + calls OpenAI chat model
- Embedding Cache Adapter (`policy_app/storage/embedding_cache.py`)
  - SQLite cache read/write

### Main query flow (number these arrows in Figma)
1. API Layer -> RAG Pipeline: "submit question with top_k and alpha"
2. RAG Pipeline -> Dense Retriever: "retrieve semantic hits"
3. Dense Retriever -> Embedding Service: "embed user question"
4. Embedding Service -> Embedding Cache Adapter: "lookup/store embeddings"
5. Embedding Service -> OpenAI API: "request embeddings for misses"
6. RAG Pipeline -> Lexical Retriever: "retrieve lexical hits"
7. RAG Pipeline -> Hybrid Ranker: "merge and rank hits"
8. RAG Pipeline -> Answer Generator: "generate grounded answer from evidence"
9. Answer Generator -> OpenAI API: "chat completion with evidence packet"
10. Answer Generator -> API Layer: "return answer + cited sources"

### Ingestion flow (secondary arrows)
1. API Layer -> Data Pipeline: "ingest seed/PDF input"
2. Data Pipeline -> Loaders: "load + clean + chunk documents"
3. Data Pipeline -> Index Builder: "build dense and lexical indexes"
4. Index Builder -> Embedding Service: "embed chunks for dense index"
5. Embedding Service -> Embedding Cache Adapter/OpenAI API: "cache-aware embedding fetch"
6. Data Pipeline -> API Layer: "return PipelineData"
7. API Layer -> In-Memory Session Store: "save/extend session pipeline"

---

## 5) Figma Drawing Conventions (Simple + Professional)

Use these consistently:

- Person: stick figure or rounded card
- Software system boundary: large frame titled `Policy RAG App`
- Containers: medium rounded rectangles
- Components: small rounded rectangles
- External systems: gray rectangles outside boundary
- Data stores: cylinder shape (or rectangle with `[Data Store]` tag)
- Arrow text: always verb phrase, 3-8 words
- Keep left-to-right flow where possible

Suggested color legend:
- UI: blue
- Backend orchestration: green
- Retrieval/LLM components: orange
- Data stores: gray
- External systems: dark gray

---

## 6) GitHub-Ready Export Checklist

Before exporting from Figma:

1. Add title with level and date, e.g. `C4 Level 2 - Policy RAG App - 2026-03-01`
2. Add small legend (colors + shape meaning)
3. Ensure text is readable at 100% zoom
4. Keep arrow crossings minimal
5. Export PNG (2x) for README
6. Save source `.fig` for future updates

---

## 7) Scope Notes (So You Stay Accurate)

- Current runtime session state is in-memory in FastAPI (`STATE`), not Redis/Postgres.
- Embedding cache is SQLite-based and optional via settings.
- Retrieval is hybrid: dense + lexical fused by weighted merge (`alpha`).
- Answers are citation-aware and based on retrieved evidence chunks.
