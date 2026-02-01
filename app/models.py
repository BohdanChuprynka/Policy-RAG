from calendar import c
from typing import Optional, List, Dict
from pydantic import BaseModel



class Chunk(BaseModel):
      chunk_id: str
      text: str
      doc_name: str
      page: Optional[int]
      
      
class DenseIndexMeta(BaseModel):
      chunk_ids: List[str]

# TODO: Create LexicalIndex model:
# - tokenized_docs: List[List[str]]   (tokens per chunk)
# - doc_freq: Dict[str, int]          (token -> doc frequency)
# - avgdl: float                      (average doc length)
# - chunk_ids: List[str]
class LexicalIndex(BaseModel): 
      tokenized_docs: List[List[str]]
      doc_freq: Dict[str, int]
      avg_doc_length: float 
      chunk_ids: List[str]

# TODO: Create RetrievalHit model:
# - chunk_id: str
# - score: float

# (Later, for generation)
# TODO: Create SourceRef model:
# - doc_name: str
# - page: Optional[int]
# - snippet: str

# TODO: Create QueryResult model:
# - answer: str
# - sources: List[SourceRef]
# - num_contexts: int