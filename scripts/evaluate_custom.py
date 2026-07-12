"""Lightweight custom RAG evaluation — no RAGAS dependency.

Uses:
- context relevance: already computed by rag_engine (cosine similarity, no extra calls)
- answer relevancy: cosine similarity between question and answer embeddings (local, no API call)
- faithfulness: ONE LLM-as-judge call per question (not per-statement like RAGAS), asking
  whether the answer is fully supported by the retrieved context
- refusal accuracy: same substring check as before, for negative/out-of-scope questions

Total Groq calls: ~1 (ask) + 1 (judge) per positive question, + 1 per negative question.
For 20 questions, that's roughly 37 calls total, vs. 300+ with the RAGAS approach.
"""

import json
import os
import re
import sys
import time

import numpy as np
from dotenv import load_dotenv
from groq import RateLimitError
from langchain_core.prompts import PromptTemplate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_engine import ask, get_embedding_model, get_llm

load_dotenv()

DATASET_PATH = "data_eval/golden_qa_dataset.json"
RESULTS_PATH = "data_eval/evaluation_results.md"
SECONDS_BETWEEN_CALLS = 6
MAX_RETRIES = 4

FAITHFULNESS_PROMPT = PromptTemplate(
    template="""You are grading whether an AI answer is fully supported by the given context.

Context:
{context}

Answer:
{answer}

On a scale of 0.0 to 1.0, how well is every claim in the answer supported by the context?
1.0 means every claim is directly supported. 0.0 means the answer contains claims not found
in the context at all. Respond with ONLY a number between 0.0 and 1.0, nothing else.""",
    input_variables=["context", "answer"],
)


def call_with_retry(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            wait = 15 * attempt
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Giving up after retries")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score_faithfulness(context: str, answer: str, llm) -> float:
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    response = call_with_retry(llm.invoke, prompt)
    match = re.search(r"(\d*\.?\d+)", response.content)
    return float(match.group(1)) if match else 0.0


def main() -> None:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)["dataset"]

    positive = [d for d in dataset if d["type"] == "positive"]
    negative = [d for d in dataset if d["type"] == "negative"]

    embedder = get_embedding_model()
    llm = get_llm()

    faithfulness_scores = []
    context_relevance_scores = []
    answer_relevance_scores = []

    for i, item in enumerate(positive, start=1):
        print(f"[{i}/{len(positive)}] {item['question']}")
        result = call_with_retry(ask, item["question"])
        time.sleep(SECONDS_BETWEEN_CALLS)

        context_text = "\n\n".join(doc.page_content for doc in result["sources"])
        if result["relevance"] is not None:
            context_relevance_scores.append(result["relevance"] / 100)

        q_embedding = embedder.embed_query(item["question"])
        a_embedding = embedder.embed_query(result["answer"])
        answer_relevance_scores.append(cosine_similarity(q_embedding, a_embedding))

        if context_text:
            score = score_faithfulness(context_text, result["answer"], llm)
            faithfulness_scores.append(score)
            time.sleep(SECONDS_BETWEEN_CALLS)

    print(f"\nRunning refusal-accuracy check on {len(negative)} negative questions...")
    correct = 0
    for i, item in enumerate(negative, start=1):
        print(f"[{i}/{len(negative)}] {item['question']}")
        result = call_with_retry(ask, item["question"])
        if "not available" in result["answer"].lower():
            correct += 1
        time.sleep(SECONDS_BETWEEN_CALLS)

    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else None

    report = f"""# RAG Evaluation Results (custom lightweight harness)

| Metric | Score | Method |
|---|---|---|
| Faithfulness (avg) | {avg(faithfulness_scores)} | Single LLM-as-judge call per question |
| Context Relevance (avg) | {avg(context_relevance_scores)} | Cosine similarity, query vs. retrieved chunks |
| Answer Relevance (avg) | {avg(answer_relevance_scores)} | Cosine similarity, query vs. answer embedding |
| Refusal Accuracy (negative Qs) | {correct}/{len(negative)} | Exact substring check on "not available" |
"""

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved results to {RESULTS_PATH}")
    print(report)


if __name__ == "__main__":
    main()