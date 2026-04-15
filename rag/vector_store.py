from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DIR = "./chroma_db"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load a local sentence-transformers embedding model.

    all-MiniLM-L6-v2 converts text → 384-dimensional vectors.
    Semantically similar text ends up close together in vector space.
    No API key needed — runs entirely on your machine.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(docs: list[Document]) -> Chroma:
    """
    Embed documents and store them in ChromaDB.

    RAG Pipeline Steps 2 & 3:
    Chunks → Vectors (via embeddings) → ChromaDB (local vector store)

    Later, we query this store with:
    "Find the chunks most semantically similar to this question"
    """
    embeddings = get_embeddings()
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )


def add_to_vector_store(vector_store: Chroma, docs: list[Document]) -> Chroma:
    """Add new documents to an existing vector store."""
    vector_store.add_documents(docs)
    return vector_store
