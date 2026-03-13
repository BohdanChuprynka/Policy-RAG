import numpy as np
import pytest

from policy_app.models import (
    Chunk,
    DenseIndexMeta,
    LexicalIndex,
    PipelineData,
    QueryResult,
    RetrievalHit,
    SourceRef,
)


# ---- Chunk ----

def test_chunk_creation():
    c = Chunk(chunk_id="c1", text="hello world", doc_name="doc.pdf", page=3)
    assert c.chunk_id == "c1"
    assert c.text == "hello world"
    assert c.doc_name == "doc.pdf"
    assert c.page == 3


def test_chunk_page_optional():
    c = Chunk(chunk_id="c2", text="no page", doc_name="seed.txt", page=None)
    assert c.page is None


def test_chunk_roundtrip_dict():
    c = Chunk(chunk_id="c1", text="x", doc_name="d.pdf", page=1)
    d = c.model_dump()
    c2 = Chunk(**d)
    assert c == c2


# ---- DenseIndexMeta ----

def test_dense_index_meta():
    meta = DenseIndexMeta(chunk_ids=["a", "b", "c"])
    assert len(meta.chunk_ids) == 3


def test_dense_index_meta_empty():
    meta = DenseIndexMeta(chunk_ids=[])
    assert meta.chunk_ids == []


# ---- LexicalIndex ----

def test_lexical_index_creation():
    lex = LexicalIndex(
        tokenized_docs=[["hello", "world"], ["foo"]],
        doc_freq={"hello": 1, "world": 1, "foo": 1},
        avg_doc_length=1.5,
        chunk_ids=["c1", "c2"],
    )
    assert len(lex.tokenized_docs) == 2
    assert lex.avg_doc_length == 1.5


def test_lexical_index_roundtrip():
    lex = LexicalIndex(
        tokenized_docs=[["a"]],
        doc_freq={"a": 1},
        avg_doc_length=1.0,
        chunk_ids=["c1"],
    )
    d = lex.model_dump()
    lex2 = LexicalIndex(**d)
    assert lex == lex2


# ---- RetrievalHit ----

def test_retrieval_hit_ordering():
    hits = [
        RetrievalHit(chunk_id="a", score=0.3),
        RetrievalHit(chunk_id="b", score=0.9),
        RetrievalHit(chunk_id="c", score=0.6),
    ]
    ranked = sorted(hits, key=lambda h: h.score, reverse=True)
    assert [h.chunk_id for h in ranked] == ["b", "c", "a"]


# ---- SourceRef / QueryResult ----

def test_query_result_structure():
    result = QueryResult(
        answer="The answer is 42.",
        sources=[
            SourceRef(doc_name="doc.pdf", page=1, snippet="evidence here"),
            SourceRef(doc_name="doc2.pdf", page=None, snippet="more evidence"),
        ],
        num_contexts=2,
    )
    assert result.num_contexts == 2
    assert len(result.sources) == 2
    assert result.sources[1].page is None


# ---- PipelineData ----

def test_pipeline_data_accepts_numpy():
    pipeline = PipelineData(
        chunks=[Chunk(chunk_id="c1", text="t", doc_name="d", page=1)],
        dense_meta=DenseIndexMeta(chunk_ids=["c1"]),
        dense_matrix=np.zeros((1, 3), dtype=np.float32),
        lexical=LexicalIndex(
            tokenized_docs=[["t"]],
            doc_freq={"t": 1},
            avg_doc_length=1.0,
            chunk_ids=["c1"],
        ),
    )
    assert pipeline.dense_matrix.shape == (1, 3)
