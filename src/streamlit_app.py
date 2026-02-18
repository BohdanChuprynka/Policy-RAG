"""
streamlit_app.py

Goal:
- A thin Streamlit client for your FastAPI Policy RAG backend.
- No OpenAI keys here. No indexing here. UI only.

Security goals:
- Never expose OPENAI_API_KEY in Streamlit.
- Only call backend endpoints.
- Upload endpoint must require a secret token (admin-only) OR be hidden in public demo mode.

Config via env vars:
- API_BASE_URL (e.g., https://your-backend.onrender.com)
- DEMO_MODE (true/false) -> controls whether upload UI is shown
- INGEST_TOKEN (optional) -> only set locally for you; DO NOT set in public Streamlit if you want uploads disabled
"""

from __future__ import annotations

import os
import requests
import streamlit as st

# TODO: read env vars
# API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
# DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
# INGEST_TOKEN = os.getenv("INGEST_TOKEN")  # optional; if not present, don't show upload

# TODO: configure Streamlit page
# st.set_page_config(...)

# -------------------------
# HTTP helpers
# -------------------------
# TODO: implement api_get_health()
# - GET /health
# - return JSON or raise nice error

# TODO: implement api_query(question, top_k, alpha)
# - POST /query (params or JSON body depending on your API)
# - return QueryResult JSON
# - handle HTTP errors and show user-friendly messages

# TODO: implement api_ingest_pdf(uploaded_file)
# - POST /ingest/pdf
# - send multipart/form-data
# - include header X-INGEST-TOKEN if INGEST_TOKEN is set
# - handle 403 by telling user "uploads disabled in public demo"

# -------------------------
# UI layout
# -------------------------
# TODO: title + description
# - explain it's a portfolio demo
# - emphasize that it answers only from seeded policy
# - show a disclaimer (not legal advice)

# Sidebar:
# TODO: show backend status:
# - call /health on button click or on load
# - show num_chunks + has_pipeline

# Sidebar knobs:
# TODO: sliders for alpha and top_k
# - NOTE: enforce same caps as backend (e.g., max top_k=10)

# Upload UI:
# TODO: only show if (not DEMO_MODE) OR (INGEST_TOKEN is set)
# - file_uploader for pdf
# - "Ingest" button
# - spinner while uploading
# - show returned added_chunks / total_chunks

# Main QA UI:
# TODO: question input + Ask button
# - call api_query
# - display answer
# - display sources (doc_name/page/snippet) in expanders

# UX polish:
# TODO:
# - include example questions buttons
# - show warnings if backend unreachable
# - show a small footer with links to GitHub repo