import os
import time
from functools import lru_cache
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import PROMPT_TEMPLATE, SETTINGS
from safety import CRISIS_RESPONSE, is_crisis_query


class RagResult(TypedDict):
    answer: str
    sources: list[Document]
    response_time: float
    num_chunks: int
    relevance: float | None
    input_tokens: int | None
    output_tokens: int | None
    is_crisis_response: bool


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=SETTINGS.embedding_model_name)


@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing from .env")

    pc = Pinecone(api_key=api_key)
    existing = [idx["name"] for idx in pc.list_indexes()]
    if SETTINGS.pinecone_index_name not in existing:
        raise RuntimeError(
            f"Pinecone index '{SETTINGS.pinecone_index_name}' not found. Run scripts/ingest.py first."
        )

    return PineconeVectorStore(
        index=pc.Index(SETTINGS.pinecone_index_name),
        embedding=get_embedding_model(),
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    return ChatGroq(
        model=SETTINGS.groq_model_name,
        temperature=SETTINGS.temperature,
        max_tokens=SETTINGS.max_tokens,
        api_key=api_key,
    )


def get_retriever(k: int | None = None, allowed_sources: list[str] | None = None):
    search_kwargs = {
        "k": k or SETTINGS.retrieval_k,
        "fetch_k": SETTINGS.retrieval_fetch_k,
        "lambda_mult": SETTINGS.mmr_lambda,
    }
    if allowed_sources:
        search_kwargs["filter"] = {"uploaded_by": {"$in": allowed_sources}}

    return get_vectorstore().as_retriever(search_type="mmr", search_kwargs=search_kwargs)


def get_rag_chain(k: int | None = None, allowed_sources: list[str] | None = None):
    retriever = get_retriever(k=k, allowed_sources=allowed_sources)
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    llm = get_llm()

    answer_chain = (
        {"context": lambda x: format_docs(x["context"]), "question": lambda x: x["question"]}
        | prompt
        | llm
    )

    return RunnablePassthrough.assign(context=lambda x: retriever.invoke(x["question"])).assign(
        llm_message=answer_chain
    )


def _estimate_relevance(
    question: str, k: int | None = None, allowed_sources: list[str] | None = None
) -> float | None:
    try:
        kwargs = {"k": k or SETTINGS.retrieval_k}
        if allowed_sources:
            kwargs["filter"] = {"uploaded_by": {"$in": allowed_sources}}
        scored = get_vectorstore().similarity_search_with_relevance_scores(question, **kwargs)
        if not scored:
            return None
        avg = float(sum(float(score) for _, score in scored) / len(scored))
        return round(max(0.0, min(1.0, avg)) * 100, 1)
    except Exception:
        return None


def _dedupe_sources(docs: list[Document]) -> list[Document]:
    seen = set()
    deduped = []
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    return deduped


def ask(question: str, k: int | None = None, allowed_sources: list[str] | None = None) -> RagResult:
    # Checked before retrieval, on the raw question — a crisis query can still
    # score high on topical relevance (matches palliative-care content etc.),
    # so relevance alone can't be trusted to catch it.
    if is_crisis_query(question):
        return RagResult(
            answer=CRISIS_RESPONSE,
            sources=[],
            response_time=0.0,
            num_chunks=0,
            relevance=None,
            input_tokens=None,
            output_tokens=None,
            is_crisis_response=True,
        )

    relevance = _estimate_relevance(question, k=k, allowed_sources=allowed_sources)

    if relevance is not None and relevance < SETTINGS.min_relevance_threshold:
        return RagResult(
            answer="## Summary\nThis information is not available in the provided medical documents.",
            sources=[],
            response_time=0.0,
            num_chunks=0,
            relevance=relevance,
            input_tokens=None,
            output_tokens=None,
            is_crisis_response=False,
        )

    chain = get_rag_chain(k=k, allowed_sources=allowed_sources)

    start = time.perf_counter()
    response = chain.invoke({"question": question})
    elapsed = time.perf_counter() - start

    llm_message = response["llm_message"]
    usage = getattr(llm_message, "usage_metadata", None) or {}
    deduped_sources = _dedupe_sources(response["context"])

    return RagResult(
        answer=llm_message.content,
        sources=deduped_sources,
        response_time=round(elapsed, 2),
        num_chunks=len(deduped_sources),
        relevance=relevance,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        is_crisis_response=False,
    )