from typing import List, Optional
import re

from openai import AsyncOpenAI

from policy_app.config import settings
from policy_app.models import Chunk, QueryResult, SourceRef

client = AsyncOpenAI()

_CIT_RE = re.compile(r"\[(\d+)\]") 

def make_context(chunks: List[Chunk]) -> str:
    """
    Formats chunks into a readable evidence packet with stable source IDs.
    The ID order MUST match the list order so we can map citations back.
    """
    if not chunks:
        return "NO EVIDENCE PROVIDED."

    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        page = c.page if c.page is not None else "unknown"
        parts.append(
            f"[{i}]\n"
            f"Document: {c.doc_name}\n"
            f"Page: {page}\n"
            f'Content:\n"""\n{c.text}\n"""\n'
        )

    return "\n---\n".join(parts)


def _extract_cited_source_numbers(answer: str) -> List[int]:
    nums = [int(m.group(1)) for m in _CIT_RE.finditer(answer)]
    # preserve order but remove duplicates
    seen = set()
    out = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


async def answer_question(question: str, evidence: List[Chunk]) -> QueryResult:
    context = make_context(evidence)

    user_prompt = f"Question:\n{question}\n\nEvidence:\n{context}"

    resp = await client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=settings.max_tokens,
    )

    answer = (resp.choices[0].message.content or "").strip()
    if not answer:
        answer = "Not found in provided policies."

    cited_nums = _extract_cited_source_numbers(answer)

    # Map citation numbers back to chunks (1-based indexing)
    sources: List[SourceRef] = []
    for n in cited_nums:
        idx = n - 1
        if 0 <= idx < len(evidence):
            c = evidence[idx]
            snippet = c.text[:240] # for logging purposes
            sources.append(SourceRef(doc_name=c.doc_name, page=c.page, snippet=snippet))

    return QueryResult(answer=answer, sources=sources, num_contexts=len(evidence))
