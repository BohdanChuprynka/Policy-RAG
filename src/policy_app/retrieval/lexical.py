from collections import Counter
from typing import List
import math

from policy_app.models import LexicalIndex, RetrievalHit
from policy_app.utils.text import tokenize

def bm25_score(query_tokens: List[str], doc_tokens: List[str], index: LexicalIndex) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    N = len(index.tokenized_docs)
    if N == 0:
        return 0.0

    avgdl = index.avg_doc_length  
    if not avgdl or avgdl <= 0:
        avgdl = 1.0 

    k1 = 1.2                # DO NOT CHANGE
    b = 0.75                # DO NOT CHANGE
    dl = len(doc_tokens)    

    tf_counter = Counter(doc_tokens)

    score = 0.0
    for t in set(query_tokens):  
        tf = tf_counter.get(t, 0)
        if tf == 0:
            continue

        df = index.doc_freq.get(t, 0)

        idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

        denom = tf + k1 * (1 - b + b * (dl / avgdl))
        score += idf * (tf * (k1 + 1)) / denom

    return score


def lexical_retrieve(question: str, lex: LexicalIndex, top_m: int) -> List[RetrievalHit]:
    q_tokens = tokenize(question)
    hits: List[RetrievalHit] = []

    for i, chunk_id in enumerate(lex.chunk_ids):
        s = bm25_score(q_tokens, lex.tokenized_docs[i], lex)
        if s > 0:
            hits.append(RetrievalHit(chunk_id=chunk_id, score=s))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_m]