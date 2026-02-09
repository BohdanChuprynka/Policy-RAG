from typing import List, Dict
from models import RetrievalHit


def score_normalizer(hits: List[RetrievalHit]) -> List[RetrievalHit]:
    """
    Min-max normalize scores to [0, 1].
    - Empty list -> empty list
    - All scores equal -> assign 1.0 to all (all equally strong)
    """
    if not hits:
        return []

    min_score = min(h.score for h in hits)
    max_score = max(h.score for h in hits)

    if max_score == min_score:
        return [RetrievalHit(chunk_id=h.chunk_id, score=1.0) for h in hits]

    denom = max_score - min_score
    return [
        RetrievalHit(chunk_id=h.chunk_id, score=(h.score - min_score) / denom)
        for h in hits
    ]


def hybrid_merge(
    dense_hits: List[RetrievalHit],
    lex_hits: List[RetrievalHit],
    alpha: float,
    top_k: int,
) -> List[RetrievalHit]:
    """
    Weighted fusion after per-retriever normalization:
      fused = alpha * dense + (1 - alpha) * lexical
    Missing scores default to 0.
    """
    dense_norm = score_normalizer(dense_hits)
    lex_norm = score_normalizer(lex_hits)

    dense_by_id: Dict[str, float] = {h.chunk_id: h.score for h in dense_norm}
    lex_by_id: Dict[str, float] = {h.chunk_id: h.score for h in lex_norm}

    all_ids = set(dense_by_id) | set(lex_by_id)

    fused: List[RetrievalHit] = []
    for chunk_id in all_ids:
        d = dense_by_id.get(chunk_id, 0.0)
        l = lex_by_id.get(chunk_id, 0.0)
        fused_score = alpha * d + (1.0 - alpha) * l
        fused.append(RetrievalHit(chunk_id=chunk_id, score=fused_score))

    fused.sort(key=lambda h: h.score, reverse=True)
    return fused[:top_k]

# TODO: rank-based_normalization