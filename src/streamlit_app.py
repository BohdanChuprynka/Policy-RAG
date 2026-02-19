from __future__ import annotations

import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv
from policy_app.config import settings


load_dotenv()

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
INGEST_TOKEN = os.getenv("INGEST_TOKEN")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_BASE_URL
if "health" not in st.session_state:
    st.session_state.health = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

st.set_page_config(page_title="Policy RAG", page_icon=":robot_face:", layout="wide")


def _session_headers() -> dict:
    headers = {"X-Session-ID": st.session_state.session_id}
    if INGEST_TOKEN:
        headers["X-Ingest-Token"] = INGEST_TOKEN
    return headers


def _format_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        if response is not None:
            try:
                payload = response.json()
                detail = payload.get("detail") if isinstance(payload, dict) else payload
            except ValueError:
                detail = response.text
            return f"{response.status_code}: {detail}"
    return str(exc)


def api_get_health() -> dict:
    r = requests.get(f"{st.session_state.api_base_url}/health", timeout=settings.request_timeout_short)
    r.raise_for_status()
    return r.json()


def api_query(question: str, top_k: int = 6, alpha: float = 0.6) -> dict:
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
        st.session_state.health = api_get_health()
        return None
    except requests.RequestException as exc:
        st.session_state.health = None
        return _format_error(exc)


def backend_help_block(error_text: str) -> None:
    st.error(f"Backend unreachable: {error_text}")
    st.info("Start FastAPI locally and keep it running while using Streamlit.")
    st.code("uvicorn policy_app.api:app --app-dir src --reload --port 8000", language="bash")
    st.caption("If backend runs on another URL, update API Base URL in the sidebar.")


error_on_load = None
if st.session_state.health is None:
    error_on_load = refresh_health()

st.title("Policy RAG Assistant")
st.caption("Portfolio demo. Answers are grounded only in indexed policy content.")
st.warning("Disclaimer: This tool is informational only and is not legal advice.")

with st.sidebar:
    st.header("Control Panel")
    st.text_input("API Base URL", key="api_base_url")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")

    cols = st.columns(2)
    with cols[0]:
        if st.button("Refresh Health", use_container_width=True):
            error_on_load = refresh_health()
    with cols[1]:
        if st.button("New Session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.last_answer = None
            st.success("Session rotated.")

    health = st.session_state.health
    if health:
        st.success("Backend online")

    st.subheader("Retrieval")
    alpha = st.slider("Alpha (dense vs lexical)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    top_k = st.slider("Top K contexts", min_value=1, max_value=settings.max_top_k, value=6, step=1)

if error_on_load:
    backend_help_block(error_on_load)

can_upload = (not DEMO_MODE) or bool(INGEST_TOKEN)
if can_upload:
    with st.container(border=True):
        st.subheader("Upload PDF")
        st.caption("Admin upload path for adding session-scoped documents.")
        upload_file = st.file_uploader("Select a policy PDF", type=["pdf"], accept_multiple_files=False)

        if st.button("Ingest PDF", use_container_width=True, disabled=upload_file is None):
            try:
                with st.spinner("Ingesting and indexing document..."):
                    ingest_result = api_ingest_pdf(upload_file)
                st.success(
                    "Ingest complete. "
                    f"Added: {ingest_result.get('added_chunks', 0)} | "
                    f"Total: {ingest_result.get('total_chunks', 0)}"
                )
                refresh_health()
            except requests.RequestException as exc:
                st.error(f"Upload failed: {_format_error(exc)}")
else:
    st.info("Demo mode enabled. Upload is hidden.")

st.subheader("Ask a Policy Question")

example_prompts = [
    "What is this policy document about?",
    "What employee behavior is explicitly prohibited?",
    "Which safety rules are mandatory?",
]

prompt_cols = st.columns(3)
for idx, prompt in enumerate(example_prompts):
    if prompt_cols[idx].button(prompt, use_container_width=True):
        st.session_state.question_input = prompt

question = st.text_area(
    "Question",
    key="question_input",
    placeholder="Ask a question grounded in uploaded/seeded policy content...",
    height=120,
)

if st.button("Ask", type="primary", use_container_width=True, disabled=not question.strip()):
    try:
        with st.spinner("Generating answer from retrieved context..."):
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

st.divider()
st.caption(f"Built for a RAG portfolio demo. Repo: https://github.com/BohdanChuprynka")


# TODO: work on ingest token validation