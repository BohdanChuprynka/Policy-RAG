"""
app/pipeline_cli.py

Goal:
- Run your RAG system end-to-end from the terminal (no UI).
- This is your integration harness: ingest -> index -> retrieve -> merge -> generate.

Learning focus:
- How artifacts flow through the pipeline.
- Keeping IDs aligned across dense vectors, lexical docs, and chunks.
- Minimal "production shape": argument parsing, clear errors, deterministic behavior.

How you'll use it:
- python -m app.pipeline_cli --pdf McDonalds_Policy.pdf --question "..."
- or interactive mode if --question is not provided.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict

from ingest.loaders import load_seed_policy_txt, load_and_chunk_pdf

from ingest.index_build import build_dense_index, build_lexical_index

from retrieval.dense import dense_retrieve
from retrieval.lexical import lexical_retrieve
from retrieval.hybrid import hybrid_merge

from llm.generate import answer_question

from models import Chunk, DenseIndexMeta, LexicalIndex, QueryResult

import numpy as np

@dataclass
class PipelineData:
    chunks: List[Chunk]
    dense_meta: DenseIndexMeta
    dense_matrix: np.ndarray
    lexical: LexicalIndex


def rag_pipeline(seed_txt: Optional[str], pdf_path: Optional[str]) -> PipelineData:
    """
    Steps:
    1) Load chunks from seed_txt (optional) and/or pdf_path (optional)
    2) Build dense index (matrix + meta)
    3) Build lexical index
    4) Return PipelineArtifacts

    Constraints:
    - If no chunks were loaded, raise a clear ValueError.
    - Keep chunk ordering consistent: dense_meta.chunk_ids must align with dense_matrix rows,
      and lexical.chunk_ids must align with lexical.tokenized_docs entries.
    """

    if not any(seed_txt, pdf_path):
        # Raise error, should provide at least one 
        raise ValueError("Must provide either seed_txt or pdf_path")

    if seed_txt:
        chunks: List[Chunk] = load_seed_policy_txt(seed_txt)

    if pdf_path:
        if seed_txt:
            chunks.append(load_and_chunk_pdf(pdf_path))
        else:
            chunks = load_and_chunk_pdf(pdf_path) 

    if not chunks:
        raise ValueError("No chunks loaded")

    # Build dense index
    dense_matrix, dense_meta = build_dense_index(chunks)   

    # Build lexical index
    lexical = build_lexical_index(chunks)

    return PipelineData(chunks=chunks, dense_meta=dense_meta, dense_matrix=dense_matrix, lexical=lexical)   

def run_query(art: PipelineData, question: str, *, top_k: int, alpha: float):
    """
    Steps:
    1) dense_hits = dense_retrieve(question, art.dense_matrix, art.dense_meta, top_k=top_k)
    2) lex_hits   = lexical_retrieve(question, art.lexical, top_m=top_k)
    3) merged     = hybrid_merge(dense_hits, lex_hits, alpha=alpha, top_k=top_k)
    4) evidence   = map merged chunk_ids -> Chunk objects in correct order
    5) result     = answer_question(question, evidence)
    6) return result

    Note:
    - If a chunk_id is missing from lookup (shouldn't happen), skip it safely.
    """
    # TODO: build a chunk_by_id lookup dict once (chunk_id -> Chunk)
    chunk_by_id = {c.chunk_id: c for c in art.chunks}
    
    dense_hits = dense_retrieve(question, art.dense_matrix, art.dense_meta, top_k=top_k)
    lex_hits = lexical_retrieve(question, art.lexical, top_k=top_k)
    # TODO: build evidence list of Chunk objects
    merged = hybrid_merge(dense_hits, lex_hits, alpha=alpha, top_k=top_k)
    # TODO: call answer_question
    # TODO: return QueryResult
    raise NotImplementedError


def main() -> None:
    """
    CLI behavior:
    - If --question is provided: run once and print answer + sources.
    - Else: enter interactive REPL loop (user types questions).
    """
    parser = argparse.ArgumentParser(description="Policy RAG (CLI)")
    parser.add_argument("--seed-txt", type=str, default=None, help="Path to seed policy TXT")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF to ingest")
    parser.add_argument("--top-k", type=int, default=6, help="Top K chunks to use as evidence")
    parser.add_argument("--alpha", type=float, default=0.6, help="Hybrid weight for dense (0..1)")
    parser.add_argument("--question", type=str, default=None, help="Question to ask")
    args = parser.parse_args()

    # TODO: build pipeline artifacts
    # art = build_pipeline(args.seed_txt, args.pdf)

    # TODO: if args.question: run once, print formatted answer + sources, return
    # TODO: else interactive loop:
    #   - prompt user for input
    #   - call run_query
    #   - print result.answer
    #   - print sources nicely (doc_name + page)



    raise NotImplementedError
if __name__ == "__main__":
    main()