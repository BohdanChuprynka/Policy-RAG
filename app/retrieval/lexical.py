"""
retrieval/lexical.py

Goal:
- Retrieve candidate chunks using lexical similarity (BM25-like scoring).

Learning focus:
- What BM25 is doing conceptually:
  term frequency + inverse document frequency + length normalization
- Why lexical retrieval catches exact terms that embeddings may miss.
"""

from typing import List, Dict
import math

from models import LexicalIndex, RetrievalHit
from utils.text import tokenize

# TODO: Implement bm25_score(...)
# Guidance:
# - compute term frequencies for the document tokens
# - compute an IDF term using doc_freq and total number of docs
# - apply BM25 formula components (k1, b, length normalization)
# - return a float score

def term_freq(t: str, doc_tokens: List[str]) -> float:
    return doc_tokens.count(t)

def bm25_score(query_tokens: List[str], doc_tokens: List[str], index: LexicalIndex) -> float:
    term_frequencies = {t: term_freq(t, doc_tokens) for t in query_tokens}

    # Get the total number of docs 
    N = len(index.tokenized_docs)

    # Build our numerator for IDF 
    numerator = N - index.doc_freq[query_tokens[0]] + 0.5
    # Build our denominator for IDF 
    denominator = index.doc_freq[query_tokens[0]] + 0.5
    # Compute our IDF 

    idf = math.log(numerator / denominator)

    # Apply Bm25 Formula 
    k = 1.2
    b = 0.75

    score = 0
    for t, tf in term_frequencies.items():
        score += idf * (tf * (k + 1)) / (tf + k * (1 - b + b * len(doc_tokens) / index.avg_doc_length))

    return score




     
# TODO: Implement lexical_retrieve(question: str, lex: LexicalIndex, top_m: int) -> List[RetrievalHit]
# Steps:
# 1) tokenize question
# 2) score every chunk using bm25_score (yes, brute force for MVP)
# 3) keep only scores > 0
# 4) return top_m hits sorted descending
#
# Note:
# - This is not optimized; that's okay for learning and small documents.
def lexical_retrieve(question: str, lex: LexicalIndex, top_m: int) -> List[RetrievalHit]:
  tokenized = tokenize(question)

  scores = {chunk_id: bm25_score(tokenized, lex.tokenized_docs[i], lex) for i, chunk_id in enumerate(lex.chunk_ids)}
  scores = {k: v for k, v in scores.items() if v > 0}

  return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_m]


  
    
