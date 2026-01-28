from typing import List
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from app.config import EMBED_MODEL, EMBED_DIM
from utils.text import batch_items

load_dotenv()

client = OpenAI()


from typing import List
import numpy as np

def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    embeddings: List[List[float]] = []

    for batch in batch_items(texts, batch_size):
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )

        # OpenAI SDK returns an object with `.data` (list) where each item has `.embedding`
        if len(resp.data) != len(batch):
            raise ValueError(
                f"Expected {len(batch)} embeddings for this batch, got {len(resp.data)}."
            )

        embeddings.extend([item.embedding for item in resp.data])

    arr = np.array(embeddings, dtype=np.float32)

    expected_shape = (len(texts), EMBED_DIM)
    if arr.shape != expected_shape:
        raise ValueError(f"Unexpected embedding array shape {arr.shape}; expected {expected_shape}.")

    return arr

# TODO: Implement l2_normalize(mat: np.ndarray) -> np.ndarray
# - normalize each row vector
# - avoid division by zero
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    pass