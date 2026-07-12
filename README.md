# 🩺 MediBot AI

A Retrieval-Augmented Generation (RAG) medical knowledge assistant that answers **strictly** from a verified medical encyclopedia — and is designed to refuse rather than hallucinate when it doesn't know something, or when a question requires human, not AI, support.

**🔗 Live demo:** [medibot-ai-aditya.streamlit.app](https://medibot-ai-aditya.streamlit.app)
**💻 Author:** [Aditya Kumar Jha](https://www.linkedin.com/in/aditya-kumar-jha-05844b380/)

> ⚠️ **Educational / portfolio project.** Not a clinical tool, not a substitute for professional medical advice. See [Limitations](#️-data-sources--limitations) below.

---

## Why this project exists

Most RAG chatbot demos answer confidently — even when they shouldn't. MediBot AI was built around a different question: *can a RAG system be made to reliably say "I don't know" instead of improvising, and can that be measured rather than assumed?*

Two design decisions follow directly from that:

1. **A safety-first gate that runs *before* retrieval.** If a question expresses self-harm or crisis intent, the RAG pipeline is bypassed entirely — no AI-generated response, no matter how "topically relevant" retrieval might find it. The user is routed to real crisis helplines instead.
2. **An out-of-domain guard.** If retrieved context isn't actually relevant to the question, the bot says so explicitly rather than letting the LLM fill the gap with outside knowledge.

Both of these are tested, not just claimed — see [Evaluation](#-evaluation) below.

---

## ✨ Key Features

- **Retrieval-Augmented Generation** grounded in a verified medical encyclopedia (Pinecone vector store, no reliance on the LLM's own training knowledge)
- **Structured answers** — every response follows a consistent Definition / Causes / Types / Symptoms / Diagnosis / Treatment / Complications / Prevention / Summary format
- **Safety/crisis-intent gating** — regex-based pre-filter that bypasses the RAG chain entirely for self-harm-adjacent queries, returning verified crisis helpline numbers instead
- **Out-of-domain rejection** — a relevance threshold prevents the LLM from answering questions the knowledge base doesn't actually cover
- **Real observability, not decoration** — every answer shows actual relevance score, retrieved chunk count, response time, and token usage
- **Session-based conversation history** — multiple chat threads, switchable, renameable, exportable, and deletable from the sidebar
- **User-adjustable retrieval depth** — a sidebar slider controls how many chunks (`k`) are retrieved per query
- **Custom lightweight evaluation harness** — faithfulness, context relevance, answer relevance, answer correctness, and refusal accuracy, measured against a 20-question golden dataset built from the actual ingested PDF
- **Dockerized** for reproducible builds and future portability (e.g. Hugging Face Spaces)

---

## 🏗️ Architecture

```
                    ┌─────────────────┐
   User question ──▶│  Safety gate     │──▶ crisis? ──▶ Hardcoded helpline response
                    │  (safety.py)     │              (RAG pipeline bypassed entirely)
                    └────────┬────────┘
                             │ not a crisis query
                             ▼
                    ┌─────────────────┐
                    │ Relevance check  │──▶ below threshold? ──▶ "Not available" response
                    │ (cosine sim.)    │
                    └────────┬────────┘
                             │ relevant
                             ▼
                    ┌─────────────────┐
                    │  MMR Retrieval   │──▶ Pinecone (BAAI/bge-base-en-v1.5 embeddings)
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  Groq LLM        │──▶ llama-3.1-8b-instant, structured prompt
                    │  (generation)    │
                    └────────┬────────┘
                             ▼
                    Answer + sources + relevance/latency/token metadata
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | `BAAI/bge-base-en-v1.5` (via `sentence-transformers` / `langchain-huggingface`) |
| Vector database | Pinecone (serverless) |
| Orchestration | LangChain (LCEL) |
| Evaluation | Custom harness (cosine similarity + single-call LLM-as-judge) |
| Containerization | Docker, Docker Compose |
| Dependency management | [uv](https://github.com/astral-sh/uv) |

---

## 📁 Project Structure

```
medibot-ai/
├── app.py                      # Streamlit front end
├── config.py                   # Central settings + prompt template
├── rag_engine.py                # Core RAG pipeline (retrieval, generation, guards)
├── safety.py                    # Crisis-intent detection + hardcoded safe response
├── scripts/
│   ├── ingest.py                 # Builds the Pinecone index from data/*.pdf
│   ├── query_cli.py              # CLI entry point for quick testing without the UI
│   └── evaluate_custom.py        # Evaluation harness (faithfulness, relevance, correctness, refusal accuracy)
├── data/
│   └── *.pdf                     # Source medical documents
├── data_eval/
│   ├── golden_qa_dataset.json    # 20-question evaluation set (17 positive, 3 negative)
│   └── evaluation_results.md     # Latest evaluation run output
├── screenshots/                  # README screenshots
├── .streamlit/
│   └── config.toml               # Local dev settings (disables file-watcher spam)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env.example                 # Template for required environment variables
├── pyproject.toml
└── uv.lock
```

---

## 📊 Evaluation

RAGAS was tried first, but its per-metric multi-call design and version-compatibility issues made it unreliable against Groq's free-tier rate limits (300+ API calls for a 20-question set, frequent failures). It was replaced with a lightweight custom harness:

| Metric | Score | Method |
|---|---|---|
| Faithfulness (avg) | **0.80** | Single LLM-as-judge call per question — "is this answer supported by this context?" |
| Context Relevance (avg) | **0.87** | Cosine similarity, query embedding vs. retrieved chunk embeddings |
| Answer Relevance (avg) | **0.81** | Cosine similarity, query embedding vs. answer embedding |
| Answer Correctness (avg) | **0.88** | Cosine similarity, answer embedding vs. golden `ground_truth` embedding — catches answers that are internally "faithful" to a retrieved chunk but the chunk itself was the wrong one |
| Refusal Accuracy (negative Qs) | **3/3** | Exact substring check for correct refusal on 3 deliberately out-of-scope questions |

The golden dataset (`data_eval/golden_qa_dataset.json`) was built directly from the ingested PDF's actual entries — not generic assumptions — to ensure the evaluation reflects what the system can genuinely be expected to know.

Re-run the evaluation yourself:
```bash
uv run python -m scripts.evaluate_custom
```

---

## ⚠️ Data Sources & Limitations

- **Source**: *The Gale Encyclopedia of Medicine*, 2nd Edition (published ~2002). This is a reputable, editorially-reviewed consumer health reference — but it predates ~20+ years of medical developments (new treatments, current statistics, recent conditions such as COVID-19 are not covered).
- **Coverage**: only C–F range entries from this edition are currently ingested. Questions outside this range (or outside medicine generally) will correctly return "not available," not a fabricated answer.
- **Safety filter is rule-based (regex pattern matching), not exhaustive.** It catches common English phrasings of self-harm/crisis intent but is not a substitute for a dedicated crisis-detection system or human moderation. It may also over-trigger on purely educational self-harm-adjacent queries — a deliberate over-caution trade-off for a medical safety context. Hindi/Hinglish phrasing is **not currently covered**, a known gap for a India-facing deployment.
- **Answer Correctness is a semantic-similarity proxy, not a fact-checker.** Two answers can be embedding-similar while differing on a specific number or detail — this metric indicates general alignment with the reference answer, not verified factual accuracy claim-by-claim.
- **This project is not a diagnostic or clinical tool.** It is intended as a portfolio/educational demonstration of responsible RAG system design.

---

## 🚀 Getting Started (Local)

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) installed
- A [Groq](https://console.groq.com) API key
- A [Pinecone](https://www.pinecone.io) API key

### 1. Clone and install
```bash
git clone https://github.com/AdityaJha27/medibot-ai.git
cd medibot-ai
uv sync
```

### 2. Configure environment variables
```bash
cp .env.example .env
```
Edit `.env` and fill in your own keys:
```
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
```

### 3. Add source documents
Place your medical PDF(s) in the `data/` folder.

### 4. Build the vector index
```bash
uv run python -m scripts.ingest
```
This creates a Pinecone index (dimension 768, cosine metric) and embeds/upserts all chunks. Re-run this whenever `data/` changes.

### 5. Run the app
```bash
uv run streamlit run app.py
```
Visit `http://localhost:8501`.

---

## 🐳 Running with Docker

Docker is included for reproducibility and portfolio demonstration (and future compatibility with container-based platforms like Hugging Face Spaces). The live demo above is deployed via Streamlit Community Cloud directly, not this container — but the full app runs identically inside Docker.

### Prerequisites
- Docker Desktop installed and running
- Your own `.env` file (see step 2 above) — **required**, since secrets are never baked into the image

### One-time: build the Pinecone index
The container only runs the Streamlit app — ingestion needs to be run once, either locally (`uv run python -m scripts.ingest`, as above) or inside the container:
```bash
docker-compose run medibot uv run python -m scripts.ingest
```

### Run the app
```bash
docker-compose up --build
```
Visit `http://localhost:8501`.

No local Python installation or `pip install` is required — everything runs inside the container. Anyone cloning this repo with Docker Desktop installed can run the entire app with the single command above, after supplying their own `.env`.

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | API key for Groq (LLM generation) |
| `PINECONE_API_KEY` | Yes | API key for Pinecone (vector storage/retrieval) |

Never commit `.env` — it's excluded via `.gitignore` and `.dockerignore`. Use `.env.example` as a template.

---

## 🗺️ Roadmap

- [ ] Hindi/Hinglish crisis-phrase coverage for the safety filter
- [ ] Multi-source ingestion (additional encyclopedia volumes, current government health sources) with source-freshness labeling
- [ ] Per-user document upload with session-scoped, privacy-conscious storage
- [ ] Hybrid (keyword + semantic) retrieval with re-ranking

---

## 📄 License

This project is for educational and portfolio purposes.

---

## 👤 Author

**Aditya Kumar Jha**
[LinkedIn](https://www.linkedin.com/in/aditya-kumar-jha-05844b380/) · [GitHub](https://github.com/AdityaJha27)