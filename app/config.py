# --- OpenAI settings ---
EMBED_MODEL: str = "text-embedding-3-large"
EMBED_DIM: int = 3072
MODEL_NAME: str = "gpt-4o-mini"

# --- Chunking settings ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 100

# --- Retrieval settings ---
DENSE_TOPN = 5
LEX_TOPN = 5
HYBRID_TOPK = 5
ALPHA = 0.5

# --- Generation settings ---
MAX_TOKENS = 700
SYSTEM_PROMPT = (
        "You are a policy assistant.\n"
        "Rules:\n"
        "1) Answer ONLY using the evidence provided.\n"
        '2) If the evidence does not contain the answer, reply exactly: "Not found in provided policies."\n'
        "3) When you make a claim, cite the supporting evidence using bracketed source numbers like [1] or [1][3].\n"
        "4) Do not invent citations.\n"
    )


# --- Paths / storage ---
# TODO: Set upload directory name
# TODO: Set seed policy path (start with txt for simplicity)