# RAG Evaluation Results (custom lightweight harness)

| Metric | Score | Method |
|---|---|---|
| Faithfulness (avg) | 0.79 | Single LLM-as-judge call per question |
| Context Relevance (avg) | 0.87 | Cosine similarity, query vs. retrieved chunks |
| Answer Relevance (avg) | 0.81 | Cosine similarity, query vs. answer embedding |
| Refusal Accuracy (negative Qs) | 3/3 | Exact substring check on "not available" |
