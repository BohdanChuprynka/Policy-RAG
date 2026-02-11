from __future__ import annotations

from typing import Optional, List

from policy_app.ingest.loaders import load_seed_policy_txt, load_and_chunk_pdf

from policy_app.ingest.index_build import build_dense_index, build_lexical_index

from policy_app.models import Chunk, PipelineData


def data_pipeline(seed_txt: Optional[str], pdf_path: Optional[str]) -> PipelineData:
    """ 
    Accepts paths and outputs a PipelineData object with chunks required to run the RAG pipeline.
    """


    if not (seed_txt or pdf_path):
        raise ValueError("No chunks loaded")

    chunks: List[Chunk] = []

    if seed_txt:
        chunks.extend(load_seed_policy_txt(seed_txt))

    if pdf_path:
        chunks.extend(load_and_chunk_pdf(pdf_path))

    if not chunks:
        raise ValueError("No chunks loaded")

    # Build dense index
    dense_matrix, dense_meta = build_dense_index(chunks)

    # Build lexical index
    lexical = build_lexical_index(chunks)

    return PipelineData(chunks=chunks, dense_meta=dense_meta, dense_matrix=dense_matrix, lexical=lexical)


def extend_pipeline(pipeline: PipelineData, new_chunks: List[Chunk]) -> PipelineData: # used in api.py
    return PipelineData(chunks=pipeline.chunks + new_chunks, dense_meta=pipeline.dense_meta, dense_matrix=pipeline.dense_matrix, lexical=pipeline.lexical)