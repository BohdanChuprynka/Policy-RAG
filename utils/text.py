import re
from typing import List, Iterable
import numpy as np


def normalize_whitespace(text: str) -> str:
      return re.sub(r'\s+', ' ', text.strip())

# TODO: Define a simple regex-based tokenizer pattern
# - Keep it basic: letters/numbers only, lowercased
      

# TODO: Write tokenize(text: str) -> List[str]
# - normalize to lowercase
# - return list of tokens using your regex pattern

def batch_items(items: List[str], batch_size: int) -> Iterable[List[str]]:
      for i in range(0, len(items), batch_size):
            yield items[i:i+batch_size]