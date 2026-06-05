# LLM Evaluation Engineering — Working Reference

A field guide for evaluating this RAG system. Built up alongside the eval harness.

---

## The central problem

LLM outputs are unbounded text. There's no `assert output == expected`. The whole discipline of LLM evaluation is: **reduce open-ended text to a comparable score without lying about what you measured.**

Three things make this hard:

1. **No exact match.** Two correct answers can have zero token overlap. ROUGE/BLEU are lexical and miss paraphrase.
2. **Reference labels are expensive.** You need humans to write ground truth. Most teams have 30 examples, not 30,000.
3. **The judge is also a model.** Using an LLM to grade an LLM has the same failure modes as the thing it's grading.

If you do not have a metric, "improvement" is rhetoric. Eval before optimize is non-negotiable.

---

## Three eval families

| Family | Needs | Examples | When to use |
|---|---|---|---|
| **Reference-based** | Ground-truth answer | Answer correctness (semantic match), embedding cosine, ROUGE | You have human-labeled gold |
| **Reference-free** | Just input + output (+ contexts for RAG) | Faithfulness, answer relevance, coherence | Production traffic with no labels |
| **Pairwise** | Two outputs to same input | "Which is better, A or B?" — powers Chatbot Arena | Comparing model variants |

For RAG, reference-based for the small golden set + reference-free for production sampling. Both, not either.

---

## RAG decomposition — the rule that saves you

**Always evaluate the retriever and generator separately.**

A bad retriever poisons every generator metric. If your faithfulness score is low, you can't tell whether:

- the retriever returned irrelevant chunks (retriever bug), or
- the generator hallucinated despite good chunks (generator bug).

Two halves of the system, two metric sets:

| Component | Question being answered | Metrics |
|---|---|---|
| Retriever | Did we surface the right context? | **Context precision** (% of retrieved chunks that are relevant), **context recall** (% of relevant chunks we retrieved), MRR, NDCG@k |
| Generator | Given good context, did we answer well? | **Faithfulness** (every claim entailed by contexts), **answer correctness** (matches ground truth), **answer relevance** (addresses the question) |

This is the most important decomposition in RAG eval. Skipping it = flying blind.

---

## Today's four Ragas metrics

| Metric | What it asks | Range | Family | Component |
|---|---|---|---|---|
| **Faithfulness** | Is every factual claim in the answer entailed by the retrieved contexts? | 0–1 | Ref-free | Generator |
| **Answer correctness** | Does the answer match the ground-truth answer (semantically)? | 0–1 | Ref-based | End-to-end |
| **Context precision** | Of retrieved chunks, what fraction are relevant to the question? | 0–1 | Ref-based | Retriever |
| **Context recall** | Of the ground-truth answer's claims, what fraction are supported by retrieved contexts? | 0–1 | Ref-based | Retriever |

Each one is computed by an LLM-as-judge under the hood. You will implement faithfulness from scratch later — knowing the prompt that powers the metric is the difference between "uses Ragas" and "understands eval."

---

## LLM-as-judge — three traps you will hit

| Trap | What happens | Mitigation |
|---|---|---|
| **Position bias** | Judge prefers first option in pairwise prompts | Randomize order, run twice swapped, average |
| **Length bias** | Judge thinks longer = better, regardless of quality | Control for length in prompt: "ignore verbosity" + check correlation |
| **Self-preference** | GPT-4 grades GPT-4 outputs higher than equally good GPT-3.5 outputs | Use a different model family as judge, or cross-judge |

**Calibration check (mandatory before trusting a judge):** label 30–50 items by hand, measure judge–human agreement with Cohen's κ. If κ < 0.4, the judge is unreliable — fix the prompt or change the model. We will skip this today but mark it as "L3 work."

---

## Golden set design

Small and high quality beats big and noisy. 30–100 items is enough to detect 5pp regressions.

Required stratification:

- **Difficulty mix:** easy (single-chunk answer), medium (multi-chunk synthesis), hard (requires reasoning)
- **In-domain / out-of-domain:** some questions the corpus answers, some it doesn't
- **Adversarial unanswerable:** 10–20% of items should have no answer in the corpus. Best RAGs respond "I don't know" or "the policy does not specify." This catches the #1 RAG failure: confidently making things up.

Each item needs:
- `question` — the user query
- `ground_truth` — the expected answer (text)
- `ground_truth_contexts` — the passages from the source that support the answer (used for context recall)

---

## The eval ladder

| Level | What it looks like | Who's there |
|---|---|---|
| L1 | Manual vibe check, 5 questions, eyeballed | Most demos |
| L2 | Automated metrics on ad-hoc set | Most "we have evals" claims |
| L3 | Curated golden set + calibrated judge + per-component breakdown | Companies that ship LLM products |
| L4 | Eval CI gate — prompts can't merge if metric drops > X | Mature LLM teams |
| L5 | Production traffic sampling, drift detection, label feedback loop | OpenAI, Anthropic-grade |

Today targets L2 → L3 transition. L4 is the stretch goal.

---

## Three rules to internalize

1. **Eval before optimize.** Numbers come before prompt tweaks.
2. **Retriever and generator are separate models.** Always score them separately or you can't fix what you can't see.
3. **Trust but verify your judge.** Calibrate against humans, or admit your judge is a coin flip with confidence.
