"""
    Basic Pipeline CLI to test the end-to-end RAG pipeline
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List
import sys

from policy_app.models import Chunk, DenseIndexMeta, LexicalIndex, QueryResult

import numpy as np

@dataclass
class PipelineData:
    chunks: List[Chunk]
    dense_meta: DenseIndexMeta
    dense_matrix: np.ndarray
    lexical: LexicalIndex

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
