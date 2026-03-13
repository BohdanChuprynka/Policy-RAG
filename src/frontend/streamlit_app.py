from __future__ import annotations

import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

from policy_app.config import settings

load_dotenv()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = settings.api_base_url
if "health" not in st.session_state:
    st.session_state.health = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "query_history" not in st.session_state:
    st.session_state.query_history = []

st.set_page_config(page_title="Policy RAG Assistant", page_icon="📄", layout="wide")



def _session_headers() -> dict:
    return {"X-Session-ID": st.session_state.session_id}


def _format_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        if response is not None:
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                return (
                    f"{response.status_code}: Received HTML from backend URL "
                    "(likely proxy/CDN challenge page). Verify API_BASE_URL points "
                    "to the backend service and disable bot challenge for API paths."
                )
            try:
                payload = response.json()
                detail = payload.get("detail") if isinstance(payload, dict) else payload
            except ValueError:
                detail = response.text
            return f"{response.status_code}: {detail}"
    return str(exc)


@st.cache_data(ttl=120, show_spinner=False)
def api_get_health(api_base_url: str) -> dict:
    r = requests.get(f"{api_base_url}/health", timeout=settings.request_timeout_short)
    r.raise_for_status()
    return r.json()


def api_query(question: str, top_k: int, alpha: float) -> dict:
    r = requests.post(
        f"{st.session_state.api_base_url}/query",
        params={"question": question, "top_k": top_k, "alpha": alpha},
        headers=_session_headers(),
        timeout=settings.request_timeout_long,
    )
    r.raise_for_status()
    return r.json()


def api_ingest_pdf(uploaded_file) -> dict:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf",
        )
    }
    r = requests.post(
        f"{st.session_state.api_base_url}/ingest/pdf",
        files=files,
        headers=_session_headers(),
        timeout=settings.request_timeout_long,
    )
    r.raise_for_status()
    return r.json()


def refresh_health() -> str | None:
    try:
        st.session_state.health = api_get_health(st.session_state.api_base_url)
        return None
    except requests.RequestException as exc:
        st.session_state.health = None
        return _format_error(exc)


def backend_help_block(error_text: str) -> None:
    st.error(f"Backend unreachable: {error_text}")
    st.caption("Run API locally or set a deployed URL in the sidebar.")
    st.code("uvicorn policy_app.api:app --app-dir src --reload --port 8000", language="bash")


st.title("Policy RAG Assistant")
st.caption("Recruiter demo: upload policy docs, ask questions, inspect grounded sources.")

with st.sidebar:
    st.header("App Controls")
    st.text_input("API Base URL", key="api_base_url", help="FastAPI base URL")
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Check Backend", use_container_width=True):
            startup_error = refresh_health()
            if startup_error:
                st.error(startup_error)
    with action_col2:
        if st.button("New Session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.last_answer = None
            st.session_state.query_history = []
            st.success("New session created")

    health = st.session_state.health
    if health:
        st.success("Backend online")
        chunk_count = health.get("num_chunks", health.get("total_chunks", 0))
        has_seed_or_pipeline = health.get("has_pipeline", health.get("has_seed", False))
        st.metric("Indexed Chunks", chunk_count)
        st.metric("Seed Pipeline", "Ready" if has_seed_or_pipeline else "Not loaded")

    st.subheader("Retrieval Settings")
    alpha = st.slider("Alpha", min_value=0.0, max_value=1.0, value=settings.alpha, step=0.05)
    top_k = st.slider("Top K", min_value=1, max_value=settings.max_top_k, value=settings.hybrid_topk, step=1)

info_col1, info_col2, info_col3 = st.columns(3)
info_col1.info("1) Upload a PDF")
info_col2.info("2) Ask policy questions")
info_col3.info("3) Review cited source snippets")

ask_tab, ingest_tab, project_tab = st.tabs(["Ask", "Ingest", "Project Notes"])

with ask_tab:
    st.subheader("Ask a Policy Question")
    examples = [
        "What is this policy document about?",
        "What employee behavior is explicitly prohibited?",
        "Which safety rules are mandatory?",
    ]
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    if ex_col1.button(examples[0], use_container_width=True):
        st.session_state.question_input = examples[0]
    if ex_col2.button(examples[1], use_container_width=True):
        st.session_state.question_input = examples[1]
    if ex_col3.button(examples[2], use_container_width=True):
        st.session_state.question_input = examples[2]

    question = st.text_area(
        "Question",
        key="question_input",
        placeholder="Ask about policy requirements, constraints, or allowed actions...",
        height=120,
    )

    if st.button("Ask", type="primary", use_container_width=True, disabled=not question.strip()):
        try:
            with st.spinner("Retrieving evidence and generating answer..."):
                result = api_query(question.strip(), top_k=top_k, alpha=alpha)
                st.session_state.last_answer = result
                st.session_state.query_history.insert(
                    0,
                    {
                        "question": question.strip(),
                        "answer": result.get("answer", ""),
                        "num_contexts": result.get("num_contexts", 0),
                    },
                )
                st.session_state.query_history = st.session_state.query_history[:5]
        except requests.RequestException as exc:
            st.error(f"Query failed: {_format_error(exc)}")

    if st.session_state.last_answer:
        result = st.session_state.last_answer
        st.markdown("### Answer")
        st.write(result.get("answer", ""))

        sources = result.get("sources", [])
        if sources:
            st.markdown("### Sources")
            for i, source in enumerate(sources, start=1):
                doc_name = source.get("doc_name", "Unknown")
                page = source.get("page")
                label = f"[{i}] {doc_name}" if page is None else f"[{i}] {doc_name} (p.{page})"
                with st.expander(label):
                    st.write(source.get("snippet", ""))

    if st.session_state.query_history:
        st.markdown("### Recent Questions")
        for item in st.session_state.query_history:
            with st.expander(item["question"]):
                st.write(item["answer"])
                st.caption(f"Contexts used: {item['num_contexts']}")

with ingest_tab:
    st.subheader("Upload PDF")
    st.caption("Adds document chunks into the current session for retrieval.")
    upload_file = st.file_uploader("Select a policy PDF", type=["pdf"], accept_multiple_files=False)

    if st.button("Ingest PDF", use_container_width=True, disabled=upload_file is None):
        try:
            with st.spinner("Processing PDF and updating indexes..."):
                ingest_result = api_ingest_pdf(upload_file)
            st.success(
                "Ingest complete. "
                f"Added: {ingest_result.get('added_chunks', 0)} | "
                f"Total: {ingest_result.get('total_chunks', 0)}"
            )
            refresh_health()
        except requests.RequestException as exc:
            st.error(f"Upload failed: {_format_error(exc)}")

with project_tab:
    st.subheader("What This Demo Shows")
    st.markdown(
        """
        - Hybrid retrieval pipeline (dense + lexical).
        - Session-scoped ingestion for PDF policies.
        - Source-grounded answers with inspectable evidence snippets.
        - FastAPI backend + Streamlit frontend separation.
        """
    )
    st.caption(f"Repository: {settings.repo_url}")

st.divider()
