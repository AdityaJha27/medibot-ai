import time

import streamlit as st
from dotenv import load_dotenv

from rag_engine import ask, get_vectorstore

load_dotenv()

EXAMPLE_QUESTIONS = [
    "What causes high cholesterol?",
    "What is diabetic neuropathy?",
    "What does face lift surgery do?",
]

st.set_page_config(page_title="MediBot AI", page_icon="🩺", layout="wide")

st.html(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
    :root{
        --bg:#0B1220; --surface:#121A2C; --surface-2:#1A2438; --border:#23304A;
        --text:#E8EDF5; --text-muted:#8B96AC; --accent:#2DD4BF;
    }
    html, body, [class*="css"]{ font-family:'Inter',sans-serif; }
    .stApp{ background:var(--bg); color:var(--text); }
    section[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--border); }
    h1,h2,h3, .hero-title{ font-family:'Space Grotesk',sans-serif; }

    .hero{ text-align:center; padding:2.5rem 1rem 1.5rem 1rem; }
    .hero-title{ font-size:2.6rem; font-weight:700; margin-bottom:.25rem;
        background:linear-gradient(90deg, var(--text) 0%, var(--accent) 100%);
        -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
    .hero-sub{ font-size:1.05rem; color:var(--text-muted); font-weight:500; }
    .hero-tag{ font-size:.85rem; color:var(--accent); margin-top:.4rem; }

    .answer-card{ background:var(--surface); border:1px solid var(--border); border-radius:14px;
        padding:1.1rem 1.3rem; margin-top:.4rem; }
    .answer-card .label{ font-family:'JetBrains Mono',monospace; font-size:.72rem; color:var(--accent);
        text-transform:uppercase; letter-spacing:.08em; margin-bottom:.5rem; }

    .vitals-strip{ display:flex; gap:1.4rem; flex-wrap:wrap; margin-top:1rem; padding-top:.8rem;
        border-top:1px solid var(--border); font-family:'JetBrains Mono',monospace; font-size:.78rem; }
    .vital{ display:flex; align-items:center; gap:.4rem; color:var(--text-muted); }
    .vital .dot{ width:7px; height:7px; border-radius:50%; background:var(--accent); display:inline-block; }
    .vital b{ color:var(--text); font-weight:500; }

    .source-chip{ background:var(--surface-2); border:1px solid var(--border); border-radius:10px;
        padding:.6rem .8rem; margin-bottom:.5rem; }
    .source-chip .src-title{ font-weight:600; font-size:.88rem; }
    .source-chip .src-page{ font-family:'JetBrains Mono',monospace; font-size:.72rem; color:var(--accent); }

    .stButton>button{ border-radius:10px; border:1px solid var(--border); background:var(--surface-2);
        color:var(--text); }
    .stButton>button:hover{ border-color:var(--accent); color:var(--accent); }
    </style>
    """
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = 8

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}  # session_id -> {"title": str, "messages": [...]}

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None


def start_new_chat() -> None:
    # Save the current thread before wiping it, so it shows up in history.
    if st.session_state.messages and st.session_state.active_session_id:
        st.session_state.chat_sessions[st.session_state.active_session_id]["messages"] = (
            st.session_state.messages
        )
    new_id = f"session_{len(st.session_state.chat_sessions)}_{int(time.time())}"
    st.session_state.chat_sessions[new_id] = {"title": None, "messages": []}
    st.session_state.active_session_id = new_id
    st.session_state.messages = []


def switch_to_session(session_id: str) -> None:
    if st.session_state.messages and st.session_state.active_session_id:
        st.session_state.chat_sessions[st.session_state.active_session_id]["messages"] = (
            st.session_state.messages
        )
    st.session_state.active_session_id = session_id
    st.session_state.messages = st.session_state.chat_sessions[session_id]["messages"]


def run_query(question: str) -> None:
    if st.session_state.active_session_id is None:
        start_new_chat()

    st.session_state.messages.append({"role": "user", "content": question})

    # Title the session after the first question, so history shows something meaningful.
    session = st.session_state.chat_sessions[st.session_state.active_session_id]
    if session["title"] is None:
        session["title"] = question[:40] + ("…" if len(question) > 40 else "")

    with st.spinner("MediBot is reading the medical literature..."):
        try:
            result = ask(question, k=st.session_state.retrieval_k)
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "meta": result}
            )
        except Exception as e:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Error: {e}", "meta": None}
            )

    st.session_state.chat_sessions[st.session_state.active_session_id]["messages"] = (
        st.session_state.messages
    )


with st.sidebar:
    st.markdown("### 🩺 MediBot AI")

    if st.button("➕ New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.markdown("---")
    st.markdown("**Conversation History**")
    past_sessions = [
        (sid, s) for sid, s in st.session_state.chat_sessions.items() if s["messages"]
    ]
    if past_sessions:
        for sid, session in reversed(past_sessions[-10:]):
            is_active = sid == st.session_state.active_session_id
            label = ("🟢 " if is_active else "💬 ") + (session["title"] or "New chat")
            if st.button(label, key=f"hist_{sid}", use_container_width=True):
                switch_to_session(sid)
                st.rerun()
    else:
        st.caption("No conversations yet.")

    st.markdown("---")
    st.markdown("**Knowledge Base**")
    try:
        db = get_vectorstore()
        stats = db.index.describe_index_stats()
        st.caption(f"📄 Indexed vectors: `{stats.get('total_vector_count', '—')}`")
    except Exception:
        st.caption("Vectorstore not built yet — run scripts/ingest.py")

    st.markdown("---")
    st.markdown("**Retrieval settings**")
    st.session_state.retrieval_k = st.slider(
        "Chunks to retrieve (k)", min_value=2, max_value=12, value=st.session_state.retrieval_k
    )

    st.markdown("---")
    st.markdown("**Settings**")
    st.caption("Model: `llama-3.1-8b-instant` (Groq)")
    st.caption("Embeddings: `BAAI/bge-base-en-v1.5`")
    st.caption("Vector DB: Pinecone")

    st.markdown("---")
    with st.expander("About"):
        st.caption(
            "MediBot AI answers strictly from a verified medical knowledge base using RAG. "
            "It does not use outside knowledge and is not a substitute for professional medical advice."
        )

if not st.session_state.messages:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🩺 MediBot AI</div>
            <div class="hero-sub">Trusted Medical Knowledge Assistant</div>
            <div class="hero-tag">Answering only from verified medical documents using RAG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(q, use_container_width=True):
                st.session_state.pending_prompt = q

st.markdown(
    """
    <div style="background:var(--surface-2); border:1px solid var(--border); border-radius:10px;
        padding:.7rem 1rem; margin:0.5rem 0 1rem 0; font-size:.82rem; color:var(--text-muted);
        text-align:center;">
        ⚠️ Educational project — grounded in verified medical documents, not a substitute for
        professional diagnosis or care.
    </div>
    """,
    unsafe_allow_html=True,
)

for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown('<div class="answer-card"><div class="label">AI Assistant</div>', unsafe_allow_html=True)
            st.markdown(message["content"])
            st.markdown("</div>", unsafe_allow_html=True)

            meta = message.get("meta")
            if meta:
                if meta.get("is_crisis_response"):
                    st.caption(
                        "🛟 Safety response — handled outside the knowledge base, "
                        "not generated by the RAG pipeline."
                    )
                elif meta.get("num_chunks", 0) > 0:
                    st.markdown(
                        f"""
                        <div class="vitals-strip">
                            <div class="vital"><span class="dot"></span>Relevance <b>{meta['relevance']}%</b></div>
                            <div class="vital"><span class="dot"></span>Chunks <b>{meta['num_chunks']}</b></div>
                            <div class="vital"><span class="dot"></span>Time <b>{meta['response_time']}s</b></div>
                            <div class="vital"><span class="dot"></span>Tokens <b>{meta['input_tokens']}/{meta['output_tokens']}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander(f"📄 Sources ({meta['num_chunks']} retrieved chunks)"):
                        for doc in meta["sources"]:
                            source = doc.metadata.get("source", "unknown").split("/")[-1].split("\\")[-1]
                            page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
                            st.markdown(
                                f"""
                                <div class="source-chip">
                                    <div class="src-title">📄 {source}</div>
                                    <div class="src-page">Page {page}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            with st.expander("Preview paragraph", expanded=False):
                                st.caption(doc.page_content[:500] + "...")
                else:
                    st.caption("ℹ️ No matching information found in the knowledge base for this question.")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.download_button(
                        "⬇️ Export Markdown",
                        data=message["content"],
                        file_name=f"medibot_answer_{i}.md",
                        mime="text/markdown",
                        key=f"dl_{i}",
                        use_container_width=True,
                    )
                with col_b:
                    if st.button("🔁 Regenerate", key=f"regen_{i}", use_container_width=True):
                        prev_user_msg = st.session_state.messages[i - 1]["content"]
                        st.session_state.messages = st.session_state.messages[:i]
                        st.session_state.pending_prompt = prev_user_msg
                        st.rerun()

prompt = st.chat_input("Ask anything about the medical knowledge base...")
if prompt:
    st.session_state.pending_prompt = prompt

if st.session_state.pending_prompt:
    q = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    run_query(q)
    st.rerun()