from typing import List, Dict, Tuple
import numpy as np
from collections import Counter

from config import EMBED_DIM
from models import Chunk, DenseIndexMeta, LexicalIndex
from llm.embedding import embed_texts, l2_normalize
from utils.text import tokenize


def make_chunk_lookup(chunks: List[Chunk]) -> Dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}

def build_dense_index(chunks: List[Chunk]) -> Tuple[np.ndarray, DenseIndexMeta]:
    if not chunks:
        empty = np.zeros((0, EMBED_DIM), dtype=np.float32)
        return (empty, DenseIndexMeta(chunk_ids=[]))

    embeddings = embed_texts([c.text for c in chunks])
    norm_embeds = l2_normalize(embeddings)

    meta = DenseIndexMeta(chunk_ids=[c.chunk_id for c in chunks])
    return (norm_embeds, meta)

def build_lexical_index(chunks: List[Chunk]) -> LexicalIndex:
    tokenized_docs = [tokenize(c.text) for c in chunks]

    if not tokenized_docs:
        return LexicalIndex(
            tokenized_docs=[],
            doc_freq={},
            avg_doc_length=0.0,
            chunk_ids=[],
        )

    doc_freq = Counter()
    for toks in tokenized_docs:
        for t in set(toks):
            doc_freq[t] += 1

    avg_doc_length = sum(len(toks) for toks in tokenized_docs) / len(tokenized_docs)
    chunk_ids = [c.chunk_id for c in chunks]

    return LexicalIndex(
        tokenized_docs=tokenized_docs,
        doc_freq=dict(doc_freq),
        avg_doc_length=float(avg_doc_length),
        chunk_ids=chunk_ids,
    )