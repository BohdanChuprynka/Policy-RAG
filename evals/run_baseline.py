"""
LLM evaluation baseline for the Policy RAG system.

Pipeline:
  1. Build the RAG index from McDonalds_Policy.pdf (one-shot, embeddings cached on disk).
  2. For each question in golden_set.json: retrieve top-k contexts + generate an answer.
  3. Save raw RAG outputs (contexts + answers) for inspection.
  4. Score with Ragas: faithfulness, answer correctness, context precision, context recall.
  5. Score abstention behaviour on adversarial-unanswerable items with a custom metric.
  6. Write results.csv + results_summary.json.

Run from inside the eval venv:
    evals/.venv/bin/python evals/run_baseline.py

This is intentionally one file. The goal is to read top-to-bottom and understand every step.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Make src/policy_app importable when running this script from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Load OPENAI_API_KEY (etc.) from the repo .env into the process environment.
load_dotenv(REPO_ROOT / ".env")

from policy_app.pipelines.data_pipeline import data_pipeline  # noqa: E402
from policy_app.retrieval.dense import dense_retrieve  # noqa: E402
from policy_app.retrieval.lexical import lexical_retrieve  # noqa: E402
from policy_app.retrieval.hybrid import hybrid_merge  # noqa: E402
from policy_app.llm.generate import answer_question  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden_set.json"
PDF_PATH = REPO_ROOT / "data" / "McDonalds_Policy.pdf"

TOP_K = 5
ALPHA = 0.5
JUDGE_MODEL = "gpt-4o-mini"          # NOTE: same as system model — judge bias risk, see CONCEPTS.md
JUDGE_EMBED_MODEL = "text-embedding-3-small"
ABSTENTION_PHRASE = "Not found in provided policies"


# --------------------------------------------------------------------------------------
# Step 1 + 2: retrieve + generate, exposing the contexts (rag_pipeline hides them).
# --------------------------------------------------------------------------------------
async def retrieve_and_answer(art, question: str, top_k: int = TOP_K, alpha: float = ALPHA):
    """Mirror rag_pipeline.rag_pipeline but return both answer and the actual context texts."""
    chunk_by_id = {c.chunk_id: c for c in art.chunks}
    dense_hits = await dense_retrieve(question, art.dense_matrix, art.dense_meta, top_n=top_k)
    lex_hits = lexical_retrieve(question, art.lexical, top_m=top_k)
    merged = hybrid_merge(dense_hits, lex_hits, alpha=alpha, top_k=top_k)
    evidence = [chunk_by_id[h.chunk_id] for h in merged if h.chunk_id in chunk_by_id]
    qr = await answer_question(question, evidence)
    contexts = [c.text for c in evidence]
    return qr.answer, contexts


# --------------------------------------------------------------------------------------
# Step 5: custom abstention metric (Ragas does not score this directly).
# --------------------------------------------------------------------------------------
def abstention_score(response: str, expected_unanswerable: bool) -> float | None:
    """1.0 if the model abstained iff it should have, 0.0 otherwise. None for answerable items."""
    abstained = ABSTENTION_PHRASE.lower() in (response or "").lower()
    if expected_unanswerable:
        return 1.0 if abstained else 0.0
    return None  # not applicable


async def main() -> None:
    # ---------- index ----------
    print(f"[1/5] Building RAG pipeline from {PDF_PATH.name}")
    art = await data_pipeline(seed_txt=None, pdf_path=str(PDF_PATH))
    print(f"      indexed {len(art.chunks)} chunks")

    # ---------- run RAG over golden set ----------
    gs = json.loads(GOLDEN_PATH.read_text())
    items = gs["items"]
    print(f"[2/5] Running RAG over {len(items)} golden questions")

    rows = []
    for it in items:
        ans, ctxs = await retrieve_and_answer(art, it["question"])
        rows.append(
            {
                "id": it["id"],
                "user_input": it["question"],
                "response": ans,
                "retrieved_contexts": ctxs,
                "reference": it["ground_truth"],
                "_difficulty": it["difficulty"],
                "_tag": it["tag"],
                "_topic": it["topic"],
            }
        )
        print(f"      [{it['id']}] answer head: {ans[:80]!r}")

    # Save raw outputs for human inspection.
    raw_df = pd.DataFrame(rows).copy()
    raw_df["retrieved_contexts"] = raw_df["retrieved_contexts"].apply(
        lambda xs: " ||| ".join(c.replace("\n", " ")[:300] for c in xs)
    )
    raw_df.to_csv(EVAL_DIR / "rag_outputs.csv", index=False)
    print(f"      saved RAG outputs -> evals/rag_outputs.csv")

    # ---------- Ragas eval ----------
    print(f"[3/5] Scoring with Ragas (judge={JUDGE_MODEL})")
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import (
        Faithfulness,
        AnswerCorrectness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    judge = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    embedder = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=JUDGE_EMBED_MODEL))

    # Drop diagnostic fields before handing rows to Ragas (it fails on extras).
    ragas_rows = [
        {k: v for k, v in r.items() if k in ("user_input", "response", "retrieved_contexts", "reference")}
        for r in rows
    ]
    ds = EvaluationDataset.from_list(ragas_rows)

    metrics = [
        Faithfulness(llm=judge),
        AnswerCorrectness(llm=judge, embeddings=embedder),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
    ]
    result = evaluate(dataset=ds, metrics=metrics, llm=judge, embeddings=embedder, show_progress=True)

    df = result.to_pandas()
    # Decorate with diagnostic fields for per-slice analysis.
    df.insert(0, "id", [r["id"] for r in rows])
    df["_difficulty"] = [r["_difficulty"] for r in rows]
    df["_tag"] = [r["_tag"] for r in rows]
    df["_topic"] = [r["_topic"] for r in rows]

    # ---------- custom abstention metric ----------
    df["abstention_correct"] = [
        abstention_score(r["response"], r["_tag"] == "unanswerable") for r in rows
    ]

    df.to_csv(EVAL_DIR / "results.csv", index=False)
    print(f"      saved per-item results -> evals/results.csv")

    # ---------- aggregates ----------
    metric_cols = [c for c in df.columns if c not in ("id", "user_input", "response", "retrieved_contexts", "reference") and not c.startswith("_") and c != "abstention_correct"]
    agg = {col: float(df[col].mean(skipna=True)) for col in metric_cols}

    # By tag (answerable vs unanswerable).
    by_tag = {}
    for tag in df["_tag"].unique():
        slice_ = df[df["_tag"] == tag]
        by_tag[tag] = {col: float(slice_[col].mean(skipna=True)) for col in metric_cols}

    # By difficulty.
    by_diff = {}
    for d in df["_difficulty"].unique():
        slice_ = df[df["_difficulty"] == d]
        by_diff[d] = {col: float(slice_[col].mean(skipna=True)) for col in metric_cols}

    # Abstention.
    abst = df["abstention_correct"].dropna()
    abstention_summary = {
        "n_unanswerable": int(len(abst)),
        "n_correct_abstentions": int(abst.sum()),
        "abstention_accuracy": float(abst.mean()) if len(abst) else None,
    }

    summary = {
        "model": "gpt-4o-mini",
        "judge_model": JUDGE_MODEL,
        "judge_embedding_model": JUDGE_EMBED_MODEL,
        "judge_caveats": "Same model used for system + judge — self-preference bias likely. Calibrate against humans for L3.",
        "n_items": len(df),
        "top_k": TOP_K,
        "alpha": ALPHA,
        "aggregate": agg,
        "by_tag": by_tag,
        "by_difficulty": by_diff,
        "abstention": abstention_summary,
    }
    (EVAL_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print("[4/5] aggregates")
    print(json.dumps(summary, indent=2))
    print("[5/5] done")


if __name__ == "__main__":
    asyncio.run(main())
