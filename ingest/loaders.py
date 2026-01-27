import os
import uuid
from typing import List, Optional

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS
from models import Chunk
from utils.text import normalize_whitespace


splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def _safe_int_page(value: object) -> Optional[int]:
    """Best-effort conversion of common page metadata values to int.

    Handles:
      - int pages (already)
      - digit strings ("12")
    Returns None for anything else (e.g., "iv", "12 of 45").
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s)
    return None


# Takes already uploaded TXT file (our policy) and splits it into chunks
def load_seed_policy_txt(path: str) -> List[Chunk]:
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    text = normalize_whitespace(text)

    text_chunks = splitter.split_text(text)

    chunks: List[Chunk] = []
    for c in text_chunks:
        c = normalize_whitespace(c)
        if not c:
            continue
        if len(c) < MIN_CHUNK_CHARS:
            continue

        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=c,
                doc_name=os.path.basename(path),
                page=None,
            )
        )

    return chunks


# Adds additional use by allowing users to upload their own PDF files.
def load_and_chunk_pdf(file_path: str, doc_name: Optional[str] = None) -> List[Chunk]:
    reader = PDFReader()
    docs = reader.load_data(file=file_path)

    base_doc_name = doc_name or os.path.basename(file_path)

    chunks: List[Chunk] = []
    for d in docs:
        text = getattr(d, "text", None)
        if not text:
            continue

        md = getattr(d, "metadata", None) or {}
        if not isinstance(md, dict):
            md = {}

        page_raw = md.get("page") or md.get("page_label")
        page_int = _safe_int_page(page_raw)

        effective_doc_name = base_doc_name
        md_doc_name = md.get("doc_name")
        if not doc_name and isinstance(md_doc_name, str) and md_doc_name.strip():
            effective_doc_name = md_doc_name.strip()

        text_chunks = splitter.split_text(text)
        for c in text_chunks:
            c = normalize_whitespace(c)
            if not c:
                continue
            if len(c) < MIN_CHUNK_CHARS:
                continue

            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=c,
                    doc_name=effective_doc_name,
                    page=page_int,
                )
            )

    return chunks