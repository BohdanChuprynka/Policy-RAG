from typing import List
import numpy as np

from app.models import DenseIndexMeta, RetrievalHit
from app.llm.embedding import embed_texts, l2_normalize
from app.config import DENSE_TOPN


def dense_retrieve(
    question: str,
    vectors: np.ndarray,
    meta: DenseIndexMeta,
    top_n = DENSE_TOPN,
) -> List[RetrievalHit]:

    if top_n <= 0:
        return []
    if vectors.size == 0:
        return []

    q = embed_texts([question], batch_size=1)
    q = l2_normalize(q)

    scores = (vectors @ q.T).squeeze()

    # Safety: ensure 1D
    if scores.ndim != 1:
        scores = scores.reshape(-1)

    k = min(int(top_n), int(scores.shape[0]))

    top_idx = np.argsort(scores)[::-1][:k]

    return [
        RetrievalHit(chunk_id=meta.chunk_ids[i], score=float(scores[i]))
        for i in top_idx
    ]