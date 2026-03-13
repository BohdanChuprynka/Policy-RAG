from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",        
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- .env ---
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    api_base_url: str = Field("http://127.0.0.1:8000", alias="API_BASE_URL")
    repo_url: str = Field("https://github.com/BohdanChuprynka", alias="REPO_URL")
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")


    # --- OpenAI models ---
    embed_model: str = "text-embedding-3-large"
    embed_dim: int = 3072
    model_name: str = "gpt-4o-mini"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_chars: int = 100

    # --- Retrieval ---
    dense_topn: int = 5
    lex_topn: int = 5
    hybrid_topk: int = 5
    alpha: float = 0.5

    # --- Generation ---
    max_tokens: int = 700
    system_prompt: str = (
        "You are a policy assistant.\n"
        "Rules:\n"
        "1) Answer ONLY using the evidence provided.\n"
        '2) If the evidence does not contain the answer, reply exactly: "Not found in provided policies."\n'
        "3) When you make a claim, cite the supporting evidence using bracketed source numbers like [1] or [1][3].\n"
        "4) Do not invent citations.\n"
    )

    # --- Security / Limits ---

    # api.py
    session_ttl_seconds: int = 3600
    max_uploads_per_session: int = 2
    max_top_k: int = 10
    max_question_chars: int = 600
    max_upload_mb: int = 15
    allow_pdf_ingest: bool = Field(False, alias="ALLOW_PDF_INGEST")
    # streamlit_app.py
    request_timeout_short: int = 15
    request_timeout_long: int = 60
    embedding_cache_enabled: bool = Field(True, alias="EMBEDDING_CACHE_ENABLED")
    embedding_cache_path: Path | None = Field(None, alias="EMBEDDING_CACHE_PATH")

    
    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    seed_policy_txt: Path | None = Field(None, alias="SEED_POLICY_TXT")

settings = Settings()
