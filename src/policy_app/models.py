from calendar import c
from typing import Optional, List, Dict
from pydantic import BaseModel
import numpy as np


class Chunk(BaseModel):
      chunk_id: str
      text: str
      doc_name: str
      page: Optional[int]
      
class DenseIndexMeta(BaseModel):
      chunk_ids: List[str]
class LexicalIndex(BaseModel): 
      tokenized_docs: List[List[str]]
      doc_freq: Dict[str, int]
      avg_doc_length: float 
      chunk_ids: List[str]

class RetrievalHit(BaseModel):
      chunk_id: str 
      score: float

class SourceRef(BaseModel): 
      doc_name: str
      page: Optional[int] 
      snippet: str


class QueryResult(BaseModel): 
      answer: str 
      sources: List[SourceRef] 
      num_contexts: int

class PipelineData(BaseModel):
    chunks: List[Chunk]
    dense_meta: DenseIndexMeta
    dense_matrix: np.ndarray
    lexical: LexicalIndex