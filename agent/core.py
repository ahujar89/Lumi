from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma
from agent.tools import set_vector_store, search_documents, web_search


SYSTEM_PROMPT = """You are Lumi, a sharp and conversational research assistant.

When you have context provided (from documents or web search), use it to answer accurately.
When no context is provided, answer from your own knowledge.

How to respond:
- Sound natural and conversational, like a knowledgeable friend explaining something
- Use prose when it flows better, headings/bullets only when content genuinely needs structure
- Always cite your sources naturally (e.g. "According to..." or "Based on your document...")
- Synthesize and explain — don't just dump raw information"""

ROUTER_PROMPT = """You are Lumi's query router. Given the user's query and the list of uploaded documents, decide:

1. ROUTE: one of DOC_SEARCH, WEB_SEARCH, or DIRECT
   - DOC_SEARCH: user is asking about their uploaded documents or files
   - WEB_SEARCH: user needs current info, research, or facts from the internet
   - DIRECT: conversational or something answerable from general knowledge

2. DOC_NAME: if ROUTE is DOC_SEARCH and you can tell which document the user means, return its exact filename. Otherwise return ALL.

Uploaded documents: {doc_list}
Query: {query}

Reply in this exact format (two lines only):
ROUTE: <DOC_SEARCH|WEB_SEARCH|DIRECT>
DOC_NAME: <filename or ALL>"""


def create_analyst_agent(api_key: str, vector_store: Chroma = None):
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )
    set_vector_store(vector_store)
    return llm


def _classify_query(llm, query: str, uploaded_docs: list) -> tuple[str, str]:
    """
    Returns (route, doc_name_or_ALL).

    Asks the LLM to decide:
    - Which tool to use (doc search / web search / direct answer)
    - If doc search, which specific document the user is asking about
    """
    if not uploaded_docs:
        # No docs loaded — can only web search or answer directly
        prompt = f"""Classify this query: WEB_SEARCH (needs internet) or DIRECT (general knowledge).
Query: {query}
Reply with one word: WEB_SEARCH or DIRECT"""
        result = llm.invoke(prompt).content.strip().upper()
        route = "WEB_SEARCH" if "WEB" in result else "DIRECT"
        return route, "ALL"

    doc_list = ", ".join(uploaded_docs)
    prompt = ROUTER_PROMPT.format(doc_list=doc_list, query=query)
    result = llm.invoke(prompt).content.strip()

    route = "DIRECT"
    doc_name = "ALL"

    for line in result.splitlines():
        if line.startswith("ROUTE:"):
            val = line.split(":", 1)[1].strip().upper()
            if "DOC" in val:
                route = "DOC_SEARCH"
            elif "WEB" in val:
                route = "WEB_SEARCH"
        elif line.startswith("DOC_NAME:"):
            doc_name = line.split(":", 1)[1].strip()

    return route, doc_name


def stream_agent_response(
    agent,
    query: str,
    chat_history: list,
    uploaded_docs: list = None,
):
    """
    Manual agent pipeline:
    1. Classify query → which tool + which document
    2. Fetch context via the right tool (with optional doc filter)
    3. Stream the final answer
    """
    llm = agent
    uploaded_docs = uploaded_docs or []

    # Step 1: Route
    route, doc_name = _classify_query(llm, query, uploaded_docs)

    # Step 2: Fetch context
    context = ""
    if route == "DOC_SEARCH":
        yield ("status", "Looking through your documents...")
        source_filter = doc_name if doc_name != "ALL" else None
        context = search_documents(query, source_filter=source_filter)
    elif route == "WEB_SEARCH":
        yield ("status", "Searching the web...")
        context = web_search(query)

    # Step 3: Build messages
    if context:
        user_content = f"""Context retrieved:
---
{context}
---

User question: {query}

Answer using the context above. Cite sources naturally."""
    else:
        user_content = query

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_content))

    # Step 4: Stream the response
    yield ("status", None)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield ("token", chunk.content)
