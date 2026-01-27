# --- OpenAI settings ---
EMBED_MODEL: str = "text-embedding-3-large"
EMBED_DIMS: int = 3072
MODEL_NAME: str = "gpt-4o-mini"

# --- Chunking settings ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 100

# --- Retrieval settings ---
# TODO: Set dense_topN (how many dense candidates)

# TODO: Set lex_topM (how many lexical candidates)
# TODO: Set hybrid_topK (final contexts passed to generator)
# TODO: Set alpha weight for dense vs lexical in hybrid

# --- Paths / storage ---
# TODO: Set upload directory name
# TODO: Set seed policy path (start with txt for simplicity)