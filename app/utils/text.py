from __future__ import annotations

import re
from typing import List, Iterable, Tuple
import unicodedata
from collections import Counter


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def lowercase(text: str) -> str:
    return text.lower()

# def tokenizer(s: str) -> List[str]:
#     """Simple whitespace tokenizer.

#     Note: callers that need normalized comparison should use `_normalize_token`.
#     """
#     return _simple_tokenize(s)

def batch_items(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]


# --------- Below goes boilerplate related code --------- 
def tokenize(s: str) -> List[str]:
    return s.split()

def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)

def unicode_normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

_NUM_RE = re.compile(r"^\d+([.,]\d+)?$")

def _normalize_token(tok: str) -> str:
    """
    Normalization for *comparison only*:
    - lowercase
    - normalize unicode (turn fancy apostrophes into ')
    - strip leading/trailing punctuation
    - map pure numbers -> <NUM>
    """

    t = tok.lower()
    # If you normalize unicode in loaders, you can keep this off; otherwise enable it.
    # t = unicode_normalize(t)
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = t.strip(".,;:!?()[]{}<>|/\\")
    if not t:
        return ""
    if _NUM_RE.match(t):
        return "<NUM>"
    return t

def _tokenize_with_norm(s: str) -> Tuple[List[str], List[str]]:
    orig = tokenize(s)
    norm = [_normalize_token(t) for t in orig]
    return orig, norm


def _detect_frequent_start_ngrams(
    norm_texts: List[List[str]], *, n: int, min_docs_fraction: float
) -> List[Tuple[str, ...]]:
    c = Counter()
    for toks in norm_texts:
        head = [t for t in toks if t][:n]
        if len(head) == n:
            c[tuple(head)] += 1

    min_docs = max(2, int(len(norm_texts) * min_docs_fraction))
    cands = [ng for ng, cnt in c.items() if cnt >= min_docs]
    cands.sort(key=lambda ng: c[ng], reverse=True)
    return cands


def _detect_frequent_end_ngrams(
    norm_texts: List[List[str]], *, n: int, min_docs_fraction: float
) -> List[Tuple[str, ...]]:
    c = Counter()
    for toks in norm_texts:
        tail = [t for t in toks if t][-n:]
        if len(tail) == n:
            c[tuple(tail)] += 1

    min_docs = max(2, int(len(norm_texts) * min_docs_fraction))
    cands = [ng for ng, cnt in c.items() if cnt >= min_docs]
    cands.sort(key=lambda ng: c[ng], reverse=True)
    return cands


def _strip_start_boilerplate(
    orig: List[str],
    norm: List[str],
    candidates: List[Tuple[str, ...]],
    *,
    max_strip_nonempty: int,
) -> Tuple[List[str], List[str]]:
    if not candidates:
        return orig, norm

    nonempty_positions = [i for i, t in enumerate(norm) if t]
    if not nonempty_positions:
        return orig, norm

    for cand in candidates:
        k = len(cand)
        if k > max_strip_nonempty:
            continue
        if len(nonempty_positions) < k:
            continue

        head_norm = [norm[nonempty_positions[i]] for i in range(k)]
        if tuple(head_norm) == cand:
            cut_pos = nonempty_positions[k - 1] + 1
            return orig[cut_pos:], norm[cut_pos:]

    return orig, norm

def _strip_end_boilerplate(
    orig: List[str],
    norm: List[str],
    candidates: List[Tuple[str, ...]],
    *,
    max_strip_nonempty: int,
) -> Tuple[List[str], List[str]]:
    if not candidates:
        return orig, norm

    nonempty_positions = [i for i, t in enumerate(norm) if t]
    if not nonempty_positions:
        return orig, norm

    # Build list of non-empty norm tokens from the end
    for cand in candidates:
        k = len(cand)
        if k > max_strip_nonempty:
            continue
        if len(nonempty_positions) < k:
            continue

        tail_norm = [norm[nonempty_positions[-k + i]] for i in range(k)]
        if tuple(tail_norm) == cand:
            # Cut from start up to just before the first removed token
            first_removed_orig_idx = nonempty_positions[-k]
            return orig[:first_removed_orig_idx], norm[:first_removed_orig_idx]

    return orig, norm


def _adjacent_overlap_cut(
    prev_norm: List[str],
    cur_norm: List[str],
    *,
    max_overlap: int,
    min_overlap: int,
) -> int:
    prev = [t for t in prev_norm if t]
    cur = [t for t in cur_norm if t]
    kmax = min(max_overlap, len(prev), len(cur))
    for k in range(kmax, min_overlap - 1, -1):
        if prev[-k:] == cur[:k]:
            return k
    return 0


def _cut_prefix_by_nonempty_count(
    orig: List[str], norm: List[str], k_nonempty: int
) -> Tuple[List[str], List[str]]:
    if k_nonempty <= 0:
        return orig, norm
    count = 0
    cut = 0
    for i, t in enumerate(norm):
        if t:
            count += 1
        cut = i + 1
        if count >= k_nonempty:
            break
    return orig[cut:], norm[cut:]


def remove_boilerplate(texts: List[str]) -> List[str]:
    """
    Page-level cleanup:
    1) Detect & strip frequent header boilerplate (start n-grams).
    2) Detect & strip frequent footer boilerplate (end n-grams).
    3) Optionally remove duplicated overlap between adjacent pages.

    NOTE: This is token-based; it will flatten formatting.
    """
    if not texts:
        return []

    # Tunables (safer defaults for page-level)
    START_N = 10
    END_N = 10
    MIN_DOCS_FRAC = 0.70  
    MAX_STRIP = 40           
    MAX_OVERLAP = 25        
    MIN_OVERLAP = 10          

    tokenized = [_tokenize_with_norm(t) for t in texts]
    orig_list = [o for o, _ in tokenized]
    norm_list = [n for _, n in tokenized]

    start_cands = _detect_frequent_start_ngrams(
        norm_list, n=START_N, min_docs_fraction=MIN_DOCS_FRAC
    )
    end_cands = _detect_frequent_end_ngrams(
        norm_list, n=END_N, min_docs_fraction=MIN_DOCS_FRAC
    )

    # Strip header/footer boilerplate per page
    stripped_orig = []
    stripped_norm = []
    for orig, norm in zip(orig_list, norm_list):
        o2, n2 = _strip_start_boilerplate(orig, norm, start_cands, max_strip_nonempty=MAX_STRIP)
        o3, n3 = _strip_end_boilerplate(o2, n2, end_cands, max_strip_nonempty=MAX_STRIP)
        stripped_orig.append(o3)
        stripped_norm.append(n3)

    out = []
    prev_norm: List[str] = []
    for o, n in zip(stripped_orig, stripped_norm):
        if not out:
            out.append(o)
            prev_norm = n
            continue

        k = _adjacent_overlap_cut(prev_norm, n, max_overlap=MAX_OVERLAP, min_overlap=MIN_OVERLAP)
        o2, n2 = _cut_prefix_by_nonempty_count(o, n, k)
        out.append(o2)
        prev_norm = n2

    return [detokenize(o) for o in out]