import os
import sys

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_engine import ask

load_dotenv()


def main() -> None:
    user_query = input("Write Query Here: ")
    result = ask(user_query)

    print("\nANSWER:\n", result["answer"])
    print(
        f"\nRelevance: {result['relevance']}% | Chunks: {result['num_chunks']} | "
        f"Time: {result['response_time']}s | Tokens: {result['input_tokens']}/{result['output_tokens']}"
    )

    print("\nSOURCES:")
    for doc in result["sources"]:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
        print(f"- {source} (page {page}) -> {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()