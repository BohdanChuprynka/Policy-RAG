"""
    Basic Pipeline CLI to test the end-to-end RAG pipeline
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, List
import sys

from policy_app.ingest.loaders import load_seed_policy_txt, load_and_chunk_pdf

from policy_app.ingest.index_build import build_dense_index, build_lexical_index

from policy_app.retrieval.dense import dense_retrieve
from policy_app.retrieval.lexical import lexical_retrieve
from policy_app.retrieval.hybrid import hybrid_merge

from policy_app.llm.generate import answer_question

from policy_app.models import Chunk, QueryResult, PipelineData



def data_pipeline(seed_txt: Optional[str], pdf_path: Optional[str]) -> PipelineData:
    """ 
    Accepts paths and outputs a PipelineData object with chunks required to run the RAG pipeline.
    """

    if not (seed_txt and pdf_path):
        raise ValueError("No chunks loaded")

    chunks: List[Chunk] = []

    if seed_txt:
        chunks.extend(load_seed_policy_txt(seed_txt))

    if pdf_path:
        chunks.extend(load_and_chunk_pdf(pdf_path))

    if not chunks:
        raise ValueError("No chunks loaded")

    # Build dense index
    dense_matrix, dense_meta = build_dense_index(chunks)

    # Build lexical index
    lexical = build_lexical_index(chunks)

    return PipelineData(chunks=chunks, dense_meta=dense_meta, dense_matrix=dense_matrix, lexical=lexical)

def rag_pipeline(art: PipelineData, question: str, *, top_k: int, alpha: float) -> QueryResult:
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
    chunk_by_id = {c.chunk_id: c for c in art.chunks}
    
    dense_hits = dense_retrieve(question, art.dense_matrix, art.dense_meta, top_n=top_k)
    lex_hits = lexical_retrieve(question, art.lexical, top_m=top_k)
    merged = hybrid_merge(dense_hits, lex_hits, alpha=alpha, top_k=top_k)
    evidence = [chunk_by_id[h.chunk_id] for h in merged if h.chunk_id in chunk_by_id]
    query_result = answer_question(question, evidence)
    return query_result


def _format_query_result(result: QueryResult) -> str:
    lines: List[str] = []
    lines.append("Answer:")
    lines.append(result.answer.strip() if result.answer else "Not found in provided policies.")

    context_count = getattr(result, "num_contexts", None)
    if isinstance(context_count, int):
        lines.append(f"\nContexts used: {context_count}")

    lines.append("\nSources:")
    if not result.sources:
        lines.append("- None")
        return "\n".join(lines)

    for i, src in enumerate(result.sources, start=1):
        page = src.page if src.page is not None else "unknown"
        snippet = " ".join(src.snippet.split())
        lines.append(f"{i}. {src.doc_name} (page {page})")
        if snippet:
            lines.append(f"   {snippet}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy RAG (CLI)")
    parser.add_argument("--seed-txt", type=str, default=None, help="Path to seed policy TXT")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF to ingest")
    parser.add_argument("--top-k", type=int, default=6, help="Top K chunks to use as evidence")
    parser.add_argument("--alpha", type=float, default=0.6, help="Hybrid weight for dense (0..1)")
    parser.add_argument("--question", type=str, default=None, help="Question to ask")
    args = parser.parse_args()

    art = data_pipeline(args.seed_txt, args.pdf)

    if args.question:
        result = rag_pipeline(art, args.question, top_k=args.top_k, alpha=args.alpha)
        print(_format_query_result(result))
        return

    print("Entering interactive mode. Type your question (or 'exit' to quit):")
    try:
        while True:
            user_q = input("\n> ").strip()
            if user_q.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break
            if not user_q:
                continue
            result = rag_pipeline(art, user_q, top_k=args.top_k, alpha=args.alpha)
            print(_format_query_result(result))
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
