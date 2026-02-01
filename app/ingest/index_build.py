"""
ingest/index_build.py

Goal:
- Convert chunks into indexes:
  1) Dense index: embeddings matrix + meta
  2) Lexical index: tokenization + doc frequency stats
- Provide chunk lookup for citations and display.

Learning focus:
- Dense vs lexical indexing responsibilities.
- Why lexical index stores doc frequencies + avg length.
"""

from typing import List, Dict, Tuple
import numpy as np

from models import Chunk, DenseIndexMeta, LexicalIndex
from llm.embedding import embed_texts, l2_normalize
from utils.text import tokenize

# TODO: Implement make_chunk_lookup(chunks: List[Chunk]) -> Dict[str, Chunk]
# - map chunk_id to Chunk (for fast lookup during retrieval)
def make_chunk_lookup(chunks: List[Chunk]) -> Dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}

# TODO: Implement build_dense_index(chunks: List[Chunk]) -> Tuple[np.ndarray, DenseIndexMeta]
# Steps:
# 1) extract texts
# 2) embed texts
# 3) L2-normalize vectors
# 4) create DenseIndexMeta containing chunk_ids in correct order
# 5) return (vectors, meta)
def build_dense_index(chunks: List[Chunk]) -> Tuple[np.ndarray, DenseIndexMeta]:
      chunks_data = make_chunk_lookup(chunks)

      embeddings = embed_texts(chunks_data["text"]) 
      norm_embedds = l2_normalize(embeddings)

      to_return = [
            norm_embedds, 
            DenseIndexMeta(
                  chunks_data[]
            )
      ]
      



# TODO: Implement build_lexical_index(chunks: List[Chunk]) -> LexicalIndex
# Steps:
# 1) tokenize each chunk into list[str]
# 2) compute doc_freq: for each unique token in a doc, increment frequency
# 3) compute avgdl: average number of tokens per doc
# 4) store chunk_ids aligned to tokenized docs