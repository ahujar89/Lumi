# ✦ Lumi — Intelligent Research Assistant

Lumi is an AI-powered research companion built with Streamlit. You can chat with it freely, upload PDF or TXT documents and ask questions about them, or have it search the web for you — all from a single clean interface.

This project was built as a hands-on playground to learn and experiment with core AI/LLM concepts: RAG pipelines, agentic routing, vector databases, local embeddings, and streaming responses.

---

## What it does

- **Document Q&A** — Upload a PDF or TXT file. Lumi reads it, chunks it, embeds it locally, and answers questions about it with citations.
- **Web research** — Ask about anything current or factual. Lumi searches DuckDuckGo and synthesises the results.
- **Direct answers** — Conversational or general-knowledge questions are answered straight from the LLM, no retrieval needed.
- **Multi-doc support** — Upload multiple files in a session. Lumi tracks them all and can search across them or target a specific one.
- **Streaming** — Responses stream token-by-token, just like ChatGPT.

---

## AI concepts explored

### 1. RAG — Retrieval Augmented Generation
The core technique behind document Q&A. Instead of stuffing an entire document into the prompt (which would exceed context limits and be noisy), RAG breaks the document into small chunks, converts them into vectors, stores them in a database, and at query time retrieves only the most relevant chunks to feed to the LLM.

**Pipeline:**
```
PDF / TXT
    │
    ▼
Text extraction (pypdf / utf-8 decode)
    │
    ▼
Chunking  ←  chunk_size=1000 chars, overlap=200
    │
    ▼
Embedding  ←  all-MiniLM-L6-v2 (runs locally, no API key needed)
    │
    ▼
ChromaDB  ←  persisted local vector store
    │
    ▼
Similarity search at query time  →  top-4 chunks  →  LLM
```

Code: `rag/processor.py` (chunking) · `rag/vector_store.py` (embedding + storage)

---

### 2. Local Embeddings
Embeddings turn text into a list of numbers (a vector) such that semantically similar sentences end up close together in vector space. Lumi uses `all-MiniLM-L6-v2` from Sentence Transformers — a small, fast model that runs entirely on your CPU with no API calls.

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Output: 384-dimensional vectors
- Normalised embeddings → cosine similarity works out of the box

Code: `rag/vector_store.py → get_embeddings()`

---

### 3. Vector Database (ChromaDB)
ChromaDB is a local vector store. After embedding each chunk, the vectors are persisted to disk in `./chroma_db`. At query time, ChromaDB finds the nearest vectors to the query embedding (similarity search) and returns the corresponding text chunks.

Metadata (`{"source": filename}`) is stored alongside each chunk so that searches can be filtered to a specific uploaded document.

Code: `rag/vector_store.py → create_vector_store()` · `agent/tools.py → search_documents()`

---

### 4. LLM-based Query Routing (Agentic pattern)
Rather than always calling the same tool, Lumi first asks the LLM to classify the query and decide which path to take:

```
User query
    │
    ▼
Router LLM call
    ├── DOC_SEARCH  →  ChromaDB similarity search  →  answer with document context
    ├── WEB_SEARCH  →  DuckDuckGo search           →  answer with web context
    └── DIRECT      →  no retrieval                →  answer from model knowledge
```

If documents are uploaded, the router also decides *which* document the user is asking about (or `ALL` to search across everything). This avoids polluting results with irrelevant documents when the user clearly references a specific file.

Code: `agent/core.py → _classify_query()` · `agent/core.py → stream_agent_response()`

---

### 5. Streaming Responses
The final LLM call uses `.stream()` instead of `.invoke()`, yielding text chunks as they are generated. The Streamlit UI updates the response box on every token, giving a live typing effect. A status message ("Searching the web...", "Looking through your documents...") is shown while retrieval is in progress, then cleared before streaming starts.

Code: `agent/core.py → stream_agent_response()` · `app.py` (event loop)

---

### 6. Chat History + Context Window Management
The last 6 messages of the conversation are included in every request as `HumanMessage` / `AIMessage` objects so the LLM has conversational context. Capping at 6 keeps the prompt size predictable and avoids unnecessary token costs.

Code: `agent/core.py → stream_agent_response()` (messages construction)

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq API · `llama-3.3-70b-versatile` |
| LLM framework | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector store | ChromaDB (local, persisted) |
| Web search | DuckDuckGo via `langchain-community` |
| PDF parsing | pypdf |

---

## Project structure

```
analyst-ai/
├── app.py                  # Streamlit UI and session management
├── agent/
│   ├── core.py             # Query router + streaming response pipeline
│   └── tools.py            # Document search and web search tools
├── rag/
│   ├── processor.py        # File parsing and text chunking
│   └── vector_store.py     # Embedding model + ChromaDB operations
├── chroma_db/              # Local vector store (auto-created on first upload)
├── requirements.txt
└── .env.example
```

---

## Getting started

**1. Clone and install dependencies**
```bash
git clone <repo-url>
cd analyst-ai
pip install -r requirements.txt
```

**2. Set your Groq API key**

Copy `.env.example` to `.env` and add your key:
```
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com).

**3. Run**
```bash
streamlit run app.py
```

---

## How a request flows end to end

```
User types a message
        │
        ▼
_classify_query()          ← LLM decides: DOC_SEARCH / WEB_SEARCH / DIRECT
        │
   ┌────┴────────────────────┐
   │                         │
DOC_SEARCH              WEB_SEARCH           DIRECT
ChromaDB retrieval      DuckDuckGo search    (no retrieval)
top-4 similar chunks    web snippet
        │                    │                   │
        └──────────┬─────────┘                   │
                   ▼                             │
          Inject context into prompt ◄───────────┘
                   │
                   ▼
          llm.stream(messages)     ← streams tokens
                   │
                   ▼
          Streamlit updates UI token by token
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | API key for Groq (LLM inference) |
