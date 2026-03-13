import pytest

from policy_app.models import Chunk
from policy_app.ingest.index_build import build_lexical_index, make_chunk_lookup


def _chunks(texts):
    return [
        Chunk(chunk_id=f"c{i}", text=t, doc_name="doc.pdf", page=i)
        for i, t in enumerate(texts)
    ]


# ---- make_chunk_lookup ----

def test_make_chunk_lookup():
    chunks = _chunks(["a", "b"])
    lookup = make_chunk_lookup(chunks)
    assert lookup["c0"].text == "a"
    assert lookup["c1"].text == "b"


def test_make_chunk_lookup_empty():
    assert make_chunk_lookup([]) == {}


# ---- build_lexical_index ----

def test_build_lexical_index_basic():
    chunks = _chunks(["hello world", "foo bar baz"])
    index = build_lexical_index(chunks)

    assert len(index.chunk_ids) == 2
    assert len(index.tokenized_docs) == 2
    assert index.tokenized_docs[0] == ["hello", "world"]
    assert index.avg_doc_length == pytest.approx(2.5)


def test_build_lexical_index_doc_freq():
    chunks = _chunks(["hello world", "hello bar"])
    index = build_lexical_index(chunks)

    assert index.doc_freq["hello"] == 2
    assert index.doc_freq["world"] == 1
    assert index.doc_freq["bar"] == 1


def test_build_lexical_index_empty():
    index = build_lexical_index([])
    assert index.chunk_ids == []
    assert index.tokenized_docs == []
    assert index.avg_doc_length == 0.0
    assert index.doc_freq == {}
