from __future__ import annotations

import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

from policy_app.config import settings

load_dotenv()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "health" not in st.session_state:
    st.session_state.health = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

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
        f"{settings.api_base_url}/query",
        params={"question": question, "top_k": top_k, "alpha": alpha},
        headers=_session_headers(),
        timeout=settings.request_timeout_long,
    )
    r.raise_for_status()
    return r.json()


def refresh_health() -> str | None:
    try:
        st.session_state.health = api_get_health(settings.api_base_url)
        return None
    except requests.RequestException as exc:
        st.session_state.health = None
        return _format_error(exc)


st.title("Policy RAG Assistant")
st.caption("Recruiter demo: ask questions and inspect source-grounded answers.")

if not settings.api_base_url:
    st.error("`API_BASE_URL` is not configured for this cloud app.")
    st.stop()

with st.sidebar:
    st.header("Session")
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Check Backend", use_container_width=True):
            error = refresh_health()
            if error:
                st.error(error)
    with col2:
        if st.button("New Session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.last_answer = None
            st.success("New session created")

    if st.session_state.health:
        st.success("Backend online")
        chunk_count = st.session_state.health.get("num_chunks", st.session_state.health.get("total_chunks", 0))
        st.metric("Indexed Chunks", chunk_count)

    st.subheader("Retrieval")
    alpha = st.slider("Alpha", min_value=0.0, max_value=1.0, value=settings.alpha, step=0.05)
    top_k = st.slider("Top K", min_value=1, max_value=settings.max_top_k, value=settings.hybrid_topk, step=1)

st.subheader("Ask")
example_questions = [
    "What is this policy document about?",
    "What employee behavior is explicitly prohibited?",
    "Which safety rules are mandatory?",
]

b1, b2, b3 = st.columns(3)
if b1.button(example_questions[0], use_container_width=True):
    st.session_state.question_input = example_questions[0]
if b2.button(example_questions[1], use_container_width=True):
    st.session_state.question_input = example_questions[1]
if b3.button(example_questions[2], use_container_width=True):
    st.session_state.question_input = example_questions[2]

question = st.text_area(
    "Question",
    key="question_input",
    placeholder="Ask about policy requirements, constraints, or allowed actions...",
    height=120,
)

if st.button("Ask", type="primary", use_container_width=True, disabled=not question.strip()):
    try:
        with st.spinner("Retrieving evidence and generating answer..."):
            st.session_state.last_answer = api_query(question.strip(), top_k=top_k, alpha=alpha)
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

with st.expander("Admin Ingest", expanded=False):
    st.file_uploader(
        "Select a policy PDF",
        type=["pdf"],
        accept_multiple_files=False,
        disabled=True,
    )
    if st.button("Ingest PDF", use_container_width=True):
        st.info("PDF ingestion is fully implemented but disabled in the public deployment to prevent abuse and uncontrolled API costs. The complete ingestion workflow can be tested by cloning the repository and running the app locally with your own API key.")

st.divider()
st.caption(f"Portfolio demo | Repo: {settings.repo_url}")
