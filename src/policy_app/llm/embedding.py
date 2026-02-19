from typing import List
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from policy_app.config import settings
from policy_app.utils.text import batch_items

load_dotenv()

client = OpenAI()

def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, settings.embed_dim), dtype=np.float32)

    embeddings: List[List[float]] = []

    for batch in batch_items(texts, batch_size):
        resp = client.embeddings.create(
            model=settings.embed_model,
            input=batch,
        )

        # OpenAI SDK returns an object with `.data` (list) where each item has `.embedding`
        if len(resp.data) != len(batch):
            raise ValueError(
                f"Expected {len(batch)} embeddings for this batch, got {len(resp.data)}."
            )

        embeddings.extend([item.embedding for item in resp.data])

    arr = np.array(embeddings, dtype=np.float32)

    expected_shape = (len(texts), settings.embed_dim)
    if arr.shape != expected_shape:
        raise ValueError(f"Unexpected embedding array shape {arr.shape}; expected {expected_shape}.")

    return arr

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat

    arr = np.asarray(mat)

    if arr.ndim == 1:
        norm = np.linalg.norm(arr)
        if norm == 0 or not np.isfinite(norm):
            return np.zeros_like(arr)
        return (arr / norm).astype(arr.dtype, copy=False)

    if arr.ndim != 2:
        raise ValueError(f"l2_normalize expects a 1D or 2D array, got shape {arr.shape}.")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)

    # Protect against division by zero and non-finite norms.
    safe_norms = np.where((norms > 0) & np.isfinite(norms), norms, 1.0)
    out = arr / safe_norms

    # Zero-out rows that had invalid norms (0 or non-finite).
    invalid_rows = ~((norms > 0) & np.isfinite(norms))
    if np.any(invalid_rows):
        out = out.copy()
        out[invalid_rows[:, 0]] = 0.0

    return out.astype(arr.dtype, copy=False)