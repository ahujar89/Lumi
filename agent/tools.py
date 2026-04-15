from langchain_community.tools import DuckDuckGoSearchRun
from langchain_chroma import Chroma

_vector_store: Chroma = None


def set_vector_store(vs: Chroma):
    global _vector_store
    _vector_store = vs


def search_documents(query: str, source_filter: str = None) -> str:
    """
    Search uploaded documents, optionally filtered to a specific file.

    source_filter: if provided, only search chunks from that document.
    ChromaDB stores the filename in metadata["source"], so we can
    filter to a specific doc when the user is clearly asking about one.
    """
    if _vector_store is None:
        return "No documents have been uploaded yet."

    search_kwargs = {"k": 4}
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}

    retriever = _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant content found in the uploaded documents."

    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def web_search(query: str) -> str:
    search = DuckDuckGoSearchRun()
    return search.run(query)
