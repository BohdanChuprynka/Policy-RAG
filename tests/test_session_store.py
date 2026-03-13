import numpy as np
import pytest

from policy_app.models import Chunk, DenseIndexMeta, LexicalIndex, PipelineData
from policy_app.storage.session_store import _serialize, _deserialize


def _sample_pipeline(n_chunks=2, dim=4):
    chunks = [
        Chunk(chunk_id=f"c{i}", text=f"chunk text {i}", doc_name="doc.pdf", page=i)
        for i in range(n_chunks)
    ]
    return PipelineData(
        chunks=chunks,
        dense_meta=DenseIndexMeta(chunk_ids=[c.chunk_id for c in chunks]),
        dense_matrix=np.random.randn(n_chunks, dim).astype(np.float32),
        lexical=LexicalIndex(
            tokenized_docs=[["chunk", "text", str(i)] for i in range(n_chunks)],
            doc_freq={"chunk": n_chunks, "text": n_chunks},
            avg_doc_length=3.0,
            chunk_ids=[c.chunk_id for c in chunks],
        ),
    )


def test_serialize_deserialize_roundtrip():
    pipeline = _sample_pipeline()
    uploads = 1

    raw = _serialize(pipeline, uploads)
    pipeline_out, uploads_out = _deserialize(raw)

    assert uploads_out == uploads
    assert len(pipeline_out.chunks) == len(pipeline.chunks)
    for orig, restored in zip(pipeline.chunks, pipeline_out.chunks):
        assert orig.chunk_id == restored.chunk_id
        assert orig.text == restored.text
        assert orig.doc_name == restored.doc_name
        assert orig.page == restored.page


def test_serialize_preserves_dense_matrix():
    pipeline = _sample_pipeline(n_chunks=3, dim=8)
    raw = _serialize(pipeline, 0)
    restored, _ = _deserialize(raw)

    np.testing.assert_array_almost_equal(
        restored.dense_matrix, pipeline.dense_matrix, decimal=5
    )
    assert restored.dense_matrix.shape == pipeline.dense_matrix.shape


def test_serialize_preserves_lexical_index():
    pipeline = _sample_pipeline()
    raw = _serialize(pipeline, 2)
    restored, _ = _deserialize(raw)

    assert restored.lexical.tokenized_docs == pipeline.lexical.tokenized_docs
    assert restored.lexical.doc_freq == pipeline.lexical.doc_freq
    assert restored.lexical.avg_doc_length == pipeline.lexical.avg_doc_length
    assert restored.lexical.chunk_ids == pipeline.lexical.chunk_ids


def test_serialize_output_is_bytes():
    pipeline = _sample_pipeline()
    raw = _serialize(pipeline, 0)
    assert isinstance(raw, bytes)


def test_deserialize_returns_float32_matrix():
    pipeline = _sample_pipeline()
    raw = _serialize(pipeline, 0)
    restored, _ = _deserialize(raw)
    assert restored.dense_matrix.dtype == np.float32
