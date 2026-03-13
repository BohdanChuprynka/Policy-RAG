from __future__ import annotations

from policy_app.retrieval.dense import dense_retrieve
from policy_app.retrieval.lexical import lexical_retrieve
from policy_app.retrieval.hybrid import hybrid_merge

from policy_app.llm.generate import answer_question

from policy_app.models import PipelineData, QueryResult

async def rag_pipeline(art: PipelineData, question: str, *, top_k: int, alpha: float) -> QueryResult:

    chunk_by_id = {c.chunk_id: c for c in art.chunks}

    dense_hits = await dense_retrieve(question, art.dense_matrix, art.dense_meta, top_n=top_k)
    lex_hits = lexical_retrieve(question, art.lexical, top_m=top_k)
    merged = hybrid_merge(dense_hits, lex_hits, alpha=alpha, top_k=top_k)
    evidence = [chunk_by_id[h.chunk_id] for h in merged if h.chunk_id in chunk_by_id]
    query_result = await answer_question(question, evidence)
    return query_result