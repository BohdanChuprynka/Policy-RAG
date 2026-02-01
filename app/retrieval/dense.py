"""
retrieval/dense.py

Goal:
- Retrieve most relevant chunks using dense vectors (cosine similarity).

Learning focus:
- How query embeddings are used.
- Why normalized vectors make cosine similarity == dot product.
"""

from typing import List
import numpy as np

from models import DenseIndexMeta, RetrievalHit
from llm.embedding import embed_texts, l2_normalize

def dense_retrieve(question: str, vectors: np.ndarray, meta: DenseIndexMeta, top_n: int) -> List[RetrievalHit]:
      embed_question = embed_texts([question], batch_size=1) 
      norm_question = l2_normalize(embed_question)
      scores = np.dot(vectors, norm_question.T)
      top_n_indices = np.argsort(scores)[::-1][:top_n] 
      return [
            RetrievalHit(
                  chunk_id=meta.chunk_ids[i],
                  score=scores[i]) for i in top_n_indices
            ]
