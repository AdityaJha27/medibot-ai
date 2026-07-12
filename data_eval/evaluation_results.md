# RAG Evaluation Results (custom lightweight harness)

| Metric | Score | Method |
|---|---|---|
| Faithfulness (avg) | 0.8 | Single LLM-as-judge call per question |
| Context Relevance (avg) | 0.87 | Cosine similarity, query vs. retrieved chunks |
| Answer Relevance (avg) | 0.81 | Cosine similarity, query vs. answer embedding |
| Answer Correctness (avg) | 0.88 | Cosine similarity, answer vs. golden ground_truth |
| Refusal Accuracy (negative Qs) | 3/3 | Exact substring check on "not available" |
