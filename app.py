import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
from rag.processor import process_uploaded_file
from rag.vector_store import create_vector_store, add_to_vector_store
from agent.core import create_analyst_agent, stream_agent_response

load_dotenv()

st.set_page_config(
    page_title="Lumi",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Global ── */
*, *::before, *::after {
    box-sizing: border-box;
}
* {
    font-family: 'DM Sans', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── App background ── */
.stApp {
    background: #FAF7F2;
    color: #2C1F1A;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #F2EBE0 !important;
    border-right: 1px solid rgba(176, 125, 107, 0.18) !important;
}
section[data-testid="stSidebar"] > div {
    padding: 1.8rem 1.4rem !important;
}

/* ── Sidebar text ── */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #9A7B6E !important;
    font-size: 0.83rem !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(176, 125, 107, 0.15) !important;
    margin: 1rem 0 !important;
}

/* ── Buttons — shared base ── */
.stButton > button {
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 0.9rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    border: 1.5px solid !important;
}

/* Clear chat button — blush rose */
div[data-testid="column"]:first-child .stButton > button {
    background: #FDF0EE !important;
    border-color: #E8B4A8 !important;
    color: #B07060 !important;
}
div[data-testid="column"]:first-child .stButton > button:hover {
    background: #F9E4DF !important;
    border-color: #C4896A !important;
    color: #8C5040 !important;
}

/* Clear all button — warm terracotta */
div[data-testid="column"]:last-child .stButton > button {
    background: #F7EDE8 !important;
    border-color: #C4957A !important;
    color: #8C5A40 !important;
}
div[data-testid="column"]:last-child .stButton > button:hover {
    background: #F0DDD5 !important;
    border-color: #A0705A !important;
    color: #6B3A28 !important;
}

/* ── Text input (API key) ── */
.stTextInput input {
    background: #FFFCF9 !important;
    border: 1.5px solid rgba(176, 125, 107, 0.25) !important;
    border-radius: 10px !important;
    color: #2C1F1A !important;
    font-size: 0.85rem !important;
}
.stTextInput input:focus {
    border-color: #C4896A !important;
    box-shadow: 0 0 0 3px rgba(196, 137, 106, 0.1) !important;
}
.stTextInput label {
    color: #B09080 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: #FFFCF9 !important;
    border: 1.5px dashed rgba(196, 137, 106, 0.4) !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: #FDF5EF !important;
    border-color: rgba(196, 137, 106, 0.7) !important;
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span {
    color: #C0A090 !important;
    font-size: 0.8rem !important;
}

/* ── Main content area ── */
.main .block-container {
    padding: 1.5rem 2.5rem 6rem 2.5rem !important;
    max-width: 860px !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
}

/* Assistant bubble */
[data-testid="stChatMessageContent"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(176, 125, 107, 0.14) !important;
    border-radius: 4px 16px 16px 16px !important;
    padding: 0.9rem 1.15rem !important;
    color: #2C1F1A !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    box-shadow: 0 2px 16px rgba(176, 125, 107, 0.08) !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #C4896A 0%, #D4A58A 100%) !important;
    border-color: transparent !important;
    border-radius: 16px 4px 16px 16px !important;
    color: #FFF8F5 !important;
    box-shadow: 0 3px 18px rgba(196, 137, 106, 0.3) !important;
}

/* Avatars */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #C4896A, #D4A58A) !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(196, 137, 106, 0.3) !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: #FFFFFF !important;
    border: 1.5px solid rgba(196, 137, 106, 0.3) !important;
    box-shadow: 0 2px 8px rgba(176, 125, 107, 0.1) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #F2EBE0 !important;
    border-top: 1px solid rgba(176, 125, 107, 0.15) !important;
    padding: 1rem 2.5rem !important;
}
[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    border: 1.5px solid rgba(176, 125, 107, 0.2) !important;
    border-radius: 14px !important;
    color: #2C1F1A !important;
    font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #C4896A !important;
    box-shadow: 0 0 0 3px rgba(196, 137, 106, 0.1) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #C8B0A4 !important;
}

/* ── Caption (status) ── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #C0A090 !important;
    font-size: 0.78rem !important;
    font-style: italic !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #C4896A !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(196, 137, 106, 0.3);
    border-radius: 4px;
}

/* ── Code blocks ── */
code {
    background: #FDF0E8 !important;
    border: 1px solid rgba(196, 137, 106, 0.2) !important;
    border-radius: 5px !important;
    padding: 0.15em 0.4em !important;
    font-size: 0.85em !important;
    color: #A0604A !important;
}
pre {
    background: #FDF7F2 !important;
    border: 1px solid rgba(176, 125, 107, 0.15) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
pre code {
    background: transparent !important;
    border: none !important;
}

/* ── Success alert ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.83rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "vector_store": None,
    "agent": None,
    "uploaded_docs": [],
    "agent_doc_count": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Branding
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <div style="font-family: 'DM Serif Display', serif !important;
                    font-size: 1.6rem; font-weight: 400; letter-spacing: -0.01em;
                    color: #2C1F1A;">
            ✦ Lumi
        </div>
        <div style="font-size: 0.72rem; color: #C0A090; margin-top: 0.15rem;
                    letter-spacing: 0.04em; font-weight: 400;">
            your intelligent research companion
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Documents
    st.markdown("""
    <p style="font-size:0.7rem; font-weight:600; letter-spacing:0.09em;
              text-transform:uppercase; color:#C0A090; margin-bottom:0.6rem;">
        Documents
    </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a file",
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file and uploaded_file.name not in st.session_state.uploaded_docs:
        with st.spinner("Processing..."):
            docs = process_uploaded_file(uploaded_file)
            if st.session_state.vector_store is None:
                st.session_state.vector_store = create_vector_store(docs)
            else:
                add_to_vector_store(st.session_state.vector_store, docs)
            st.session_state.uploaded_docs.append(uploaded_file.name)
        st.success(f"Ready — {uploaded_file.name}")

    # Loaded doc pills
    if st.session_state.uploaded_docs:
        st.markdown('<div style="margin-top:0.6rem;">', unsafe_allow_html=True)
        for name in st.session_state.uploaded_docs:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.45rem;
                        padding:0.38rem 0.65rem;
                        background: linear-gradient(135deg, #FDF0EA 0%, #FBE8E0 100%);
                        border: 1px solid rgba(196,137,106,0.25);
                        border-radius: 20px; margin-bottom:0.3rem;">
                <span style="color:#C4896A; font-size:0.7rem;">✦</span>
                <span style="color:#8C5A40; font-size:0.78rem; font-weight:500;
                             white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
                             max-width:160px;">{name}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Capabilities
    st.markdown("""
    <p style="font-size:0.7rem; font-weight:600; letter-spacing:0.09em;
              text-transform:uppercase; color:#C0A090; margin-bottom:0.5rem;">
        What I can do
    </p>
    """, unsafe_allow_html=True)
    for cap in [
        "Analyze & summarize documents",
        "Research any topic on the web",
        "Answer from your uploaded files",
        "Blend sources for richer answers",
    ]:
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:0.4rem;
                    margin-bottom:0.3rem;">
            <span style="color:#E8B4A0; font-size:0.65rem; margin-top:0.15rem;">◆</span>
            <span style="color:#B09080; font-size:0.8rem; line-height:1.4;">{cap}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Clear all"):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.agent = None
            st.session_state.uploaded_docs = []
            st.session_state.agent_doc_count = 0
            import shutil, os as _os
            if _os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="margin-bottom: 1.5rem; padding-bottom: 1rem;
            border-bottom: 1px solid rgba(176, 125, 107, 0.12);">
    <div style="font-family: 'DM Serif Display', serif !important;
                font-size: 1.8rem; font-weight: 400; color: #2C1F1A;
                letter-spacing: -0.02em; margin-bottom: 0.15rem;">
        ✦ Lumi
    </div>
    <div style="font-size: 0.8rem; color: #C0A090; letter-spacing: 0.02em;">
        Ask anything · Research the web · Analyze your documents
    </div>
</div>
""", unsafe_allow_html=True)

# Status bar
if st.session_state.uploaded_docs:
    pills = "  ·  ".join(f"✦ {n}" for n in st.session_state.uploaded_docs)
    st.markdown(f"""
    <div style="padding:0.5rem 0.9rem; margin-bottom:1.2rem;
                background: linear-gradient(135deg, #FDF0EA 0%, #FBE8DF 100%);
                border: 1px solid rgba(196,137,106,0.22);
                border-radius: 20px; display:inline-block;">
        <span style="font-size:0.8rem; color:#B07060; font-weight:500;">{pills}</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="padding:0.5rem 0.9rem; margin-bottom:1.2rem;
                background: #F7F2EC; border: 1px solid rgba(176,125,107,0.1);
                border-radius: 20px; display:inline-block;">
        <span style="font-size:0.8rem; color:#C8B0A4;">No documents · web research mode</span>
    </div>
    """, unsafe_allow_html=True)

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:3.5rem 0 2rem; color:#D4B8AC;">
        <div style="font-family:'DM Serif Display', serif; font-size:2.8rem;
                    margin-bottom:0.6rem; opacity:0.5;">✦</div>
        <div style="font-size:0.9rem; color:#C8B0A4; line-height:1.7;">
            Ask me anything.<br>
            Upload a document or explore any topic.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input & response
if prompt := st.chat_input("Ask Lumi anything..."):
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file and restart the app.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    doc_count = len(st.session_state.uploaded_docs)
    if st.session_state.agent is None or st.session_state.agent_doc_count != doc_count:
        st.session_state.agent = create_analyst_agent(
            api_key=os.getenv("GROQ_API_KEY"),
            vector_store=st.session_state.vector_store,
        )
        st.session_state.agent_doc_count = doc_count

    with st.chat_message("assistant"):
        status_area = st.empty()
        response_box = st.empty()
        full_response = ""

        try:
            for event_type, text in stream_agent_response(
                agent=st.session_state.agent,
                query=prompt,
                chat_history=st.session_state.messages[:-1],
                uploaded_docs=st.session_state.uploaded_docs,
            ):
                if event_type == "status":
                    if text:
                        status_area.caption(text)
                    else:
                        status_area.empty()
                elif event_type == "token":
                    full_response += text
                    response_box.markdown(full_response + "▌")

            response_box.markdown(full_response)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            full_response = full_response + f"\n\n_{error_msg}_" if full_response else f"Error: {error_msg}"
            response_box.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
