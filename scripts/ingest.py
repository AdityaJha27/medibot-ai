import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SETTINGS

load_dotenv()


def load_pdf_files(data_path: str) -> list[Document]:
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def create_chunks(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
    )
    return splitter.split_documents(documents)


def ensure_index_exists(pc: Pinecone) -> None:
    existing = [idx["name"] for idx in pc.list_indexes()]
    if SETTINGS.pinecone_index_name not in existing:
        print(f"Creating index '{SETTINGS.pinecone_index_name}'...")
        pc.create_index(
            name=SETTINGS.pinecone_index_name,
            dimension=SETTINGS.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"Using existing index '{SETTINGS.pinecone_index_name}'.")


def main(uploaded_by: str = "admin") -> None:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing from .env")

    print("Loading PDFs...")
    documents = load_pdf_files(SETTINGS.data_path)
    print(f"Loaded {len(documents)} pages.")

    chunks = create_chunks(documents)
    for chunk in chunks:
        chunk.metadata["uploaded_by"] = uploaded_by
    print(f"Created {len(chunks)} chunks.")

    pc = Pinecone(api_key=api_key)
    ensure_index_exists(pc)

    print("Embedding and uploading to Pinecone...")
    embeddings_model = HuggingFaceEmbeddings(model_name=SETTINGS.embedding_model_name)
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        index_name=SETTINGS.pinecone_index_name,
    )
    print("Done.")


if __name__ == "__main__":
    main()