import pytest

from policy_app.models import LexicalIndex, RetrievalHit
from policy_app.retrieval.lexical import bm25_score, lexical_retrieve
from policy_app.retrieval.hybrid import score_normalizer, hybrid_merge


# ---- BM25 score ----

def _make_index(tokenized_docs, doc_freq=None):
    if doc_freq is None:
        from collections import Counter
        df = Counter()
        for toks in tokenized_docs:
            for t in set(toks):
                df[t] += 1
        doc_freq = dict(df)
    avg_len = (
        sum(len(t) for t in tokenized_docs) / len(tokenized_docs)
        if tokenized_docs
        else 0.0
    )
    return LexicalIndex(
        tokenized_docs=tokenized_docs,
        doc_freq=doc_freq,
        avg_doc_length=avg_len,
        chunk_ids=[f"c{i}" for i in range(len(tokenized_docs))],
    )


def test_bm25_matching_term_positive():
    docs = [["policy", "safety", "rules"], ["other", "content"]]
    index = _make_index(docs)
    score = bm25_score(["safety"], docs[0], index)
    assert score > 0


def test_bm25_no_match_returns_zero():
    docs = [["policy", "safety"]]
    index = _make_index(docs)
    score = bm25_score(["missing"], docs[0], index)
    assert score == 0.0


def test_bm25_empty_query_returns_zero():
    docs = [["word"]]
    index = _make_index(docs)
    assert bm25_score([], docs[0], index) == 0.0


def test_bm25_empty_doc_returns_zero():
    docs = [["word"]]
    index = _make_index(docs)
    assert bm25_score(["word"], [], index) == 0.0


def test_bm25_rare_term_scores_higher():
    docs = [
        ["common", "rare"],
        ["common", "other"],
        ["common", "stuff"],
    ]
    index = _make_index(docs)
    score_rare = bm25_score(["rare"], docs[0], index)
    score_common = bm25_score(["common"], docs[0], index)
    assert score_rare > score_common


# ---- lexical_retrieve ----

def test_lexical_retrieve_returns_matches():
    docs = [["safety", "rules"], ["hello", "world"], ["safety", "policy"]]
    index = _make_index(docs)
    hits = lexical_retrieve("safety", index, top_m=5)
    assert len(hits) == 2
    assert all(h.score > 0 for h in hits)


def test_lexical_retrieve_respects_top_m():
    docs = [["a", "b"], ["a", "c"], ["a", "d"]]
    index = _make_index(docs)
    hits = lexical_retrieve("a", index, top_m=2)
    assert len(hits) == 2


def test_lexical_retrieve_no_match():
    docs = [["hello"]]
    index = _make_index(docs)
    hits = lexical_retrieve("missing", index, top_m=5)
    assert hits == []


# ---- score_normalizer ----

def test_score_normalizer_empty():
    assert score_normalizer([]) == []


def test_score_normalizer_single_hit():
    hits = [RetrievalHit(chunk_id="a", score=5.0)]
    normed = score_normalizer(hits)
    assert normed[0].score == 1.0


def test_score_normalizer_range():
    hits = [
        RetrievalHit(chunk_id="a", score=1.0),
        RetrievalHit(chunk_id="b", score=3.0),
        RetrievalHit(chunk_id="c", score=5.0),
    ]
    normed = score_normalizer(hits)
    scores = {h.chunk_id: h.score for h in normed}
    assert scores["a"] == pytest.approx(0.0)
    assert scores["b"] == pytest.approx(0.5)
    assert scores["c"] == pytest.approx(1.0)


def test_score_normalizer_equal_scores():
    hits = [
        RetrievalHit(chunk_id="a", score=2.0),
        RetrievalHit(chunk_id="b", score=2.0),
    ]
    normed = score_normalizer(hits)
    assert all(h.score == 1.0 for h in normed)


# ---- hybrid_merge ----

def test_hybrid_merge_dense_only():
    dense = [RetrievalHit(chunk_id="a", score=0.9)]
    result = hybrid_merge(dense, [], alpha=1.0, top_k=5)
    assert len(result) == 1
    assert result[0].chunk_id == "a"


def test_hybrid_merge_lexical_only():
    lex = [RetrievalHit(chunk_id="b", score=0.8)]
    result = hybrid_merge([], lex, alpha=0.0, top_k=5)
    assert len(result) == 1
    assert result[0].chunk_id == "b"


def test_hybrid_merge_balanced():
    dense = [
        RetrievalHit(chunk_id="a", score=0.9),
        RetrievalHit(chunk_id="b", score=0.5),
    ]
    lex = [
        RetrievalHit(chunk_id="b", score=0.8),
        RetrievalHit(chunk_id="c", score=0.6),
    ]
    result = hybrid_merge(dense, lex, alpha=0.5, top_k=10)
    ids = [h.chunk_id for h in result]
    assert set(ids) == {"a", "b", "c"}


def test_hybrid_merge_respects_top_k():
    dense = [RetrievalHit(chunk_id=f"d{i}", score=float(i)) for i in range(10)]
    result = hybrid_merge(dense, [], alpha=1.0, top_k=3)
    assert len(result) == 3


def test_hybrid_merge_alpha_weighting():
    # With a single hit per retriever, score_normalizer maps all to 1.0.
    # So fused = alpha * 1.0 + (1 - alpha) * 1.0 = 1.0 regardless.
    # Test with two hits to exercise actual weighting.
    dense = [
        RetrievalHit(chunk_id="a", score=1.0),
        RetrievalHit(chunk_id="b", score=0.0),
    ]
    lex = [
        RetrievalHit(chunk_id="a", score=0.0),
        RetrievalHit(chunk_id="b", score=1.0),
    ]
    result = hybrid_merge(dense, lex, alpha=0.7, top_k=5)
    scores = {h.chunk_id: h.score for h in result}
    assert scores["a"] == pytest.approx(0.7)
    assert scores["b"] == pytest.approx(0.3)
