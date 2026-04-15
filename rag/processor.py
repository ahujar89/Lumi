from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
import io


def process_uploaded_file(uploaded_file) -> list[Document]:
    """
    Parse and chunk an uploaded file into LangChain Document objects.

    This is Step 1 of the RAG pipeline:
    Raw file → Text → Chunks → (next: Embed → Store in ChromaDB)
    """
    raw_text = _extract_text(uploaded_file)
    chunks = _split_into_chunks(raw_text, source=uploaded_file.name)
    return chunks


def _extract_text(uploaded_file) -> str:
    """Extract raw text from a PDF or TXT file."""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return uploaded_file.read().decode("utf-8")


def _split_into_chunks(text: str, source: str) -> list[Document]:
    """
    Split text into overlapping chunks.

    Why chunk?
    - LLMs have context window limits
    - Smaller chunks = more precise similarity search results

    chunk_size=1000  : ~1000 characters per chunk
    chunk_overlap=200: chunks overlap so context isn't lost at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source}],
    )
