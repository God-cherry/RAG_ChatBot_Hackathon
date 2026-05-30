"""
app.py  —  RAG Chatbot  (Streamlit UI)

Run this file with:
    streamlit run app.py

What the app does:
  • Sidebar  → enter API key, upload PDFs, see what's indexed
  • Main area → chat with your documents, see sources + latency
"""
import tempfile
import os
import streamlit as st

# Import our RAG engine (the other file we wrote)
from rag_engine import setup_gemini, get_collection, ingest_pdf, ask, get_stats


# =============================================================================
# Page setup  (must be the very first Streamlit command)
# =============================================================================

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚",
    layout="wide",
)


# =============================================================================
# Simple styling
# =============================================================================

st.markdown("""
<style>
/* clean dark card look */
.source-box {
    background: #1e293b;
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    color: #cbd5e1;
}
.stat-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    margin-bottom: 8px;
}
.stat-num  { font-size: 1.6rem; font-weight: 700; color: #38bdf8; }
.stat-lbl  { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session state — remembers things across re-runs
# =============================================================================

if "messages"   not in st.session_state: st.session_state.messages   = []
if "collection" not in st.session_state: st.session_state.collection = None
if "ready"      not in st.session_state: st.session_state.ready      = False


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("📚 RAG Chatbot")
    st.caption("Ask questions across your PDF library")
    st.divider()

    # ── 1. API Key ─────────────────────────────────────────────────────────────
    st.subheader("🔑 Step 1 — Gemini API Key")
    st.caption("Get a FREE key → [aistudio.google.com](https://aistudio.google.com/app/apikey)")

    api_key = st.text_input(
        "Paste your key here",
        type="password",
        placeholder="AIzaSy...",
        value=os.environ.get("GEMINI_API_KEY", ""),
    )

    if api_key:
        # Save key to environment and connect Gemini + ChromaDB
        os.environ["GEMINI_API_KEY"] = api_key
        try:
            setup_gemini(api_key)
            if st.session_state.collection is None:
                st.session_state.collection = get_collection()
            st.session_state.ready = True
            st.success("Connected!", icon="✅")
        except Exception as e:
            st.error(f"Connection failed: {e}")
            st.session_state.ready = False
    else:
        st.warning("Enter your API key to start", icon="🔑")

    st.divider()

    # ── 2. Upload PDFs ─────────────────────────────────────────────────────────
    st.subheader("📄 Step 2 — Upload PDFs")

    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.session_state.ready:
        if st.button("⚙️ Index PDFs", use_container_width=True, type="primary"):
            progress = st.progress(0, text="Starting…")
            results  = []

            for idx, uploaded_file in enumerate(uploaded_files):
                # Save uploaded file temporarily to disk
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, uploaded_file.name)

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                progress.progress(
                    (idx) / len(uploaded_files),
                    text=f"Processing {uploaded_file.name}…"
                )

                try:
                    result = ingest_pdf(temp_path, st.session_state.collection)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "filename": uploaded_file.name,
                        "status":   "failed",
                        "error":    str(e),
                    })

                os.remove(temp_path)   # clean up temp file

            progress.progress(1.0, text="Done!")

            # Show results
            for r in results:
                if r["status"] == "success":
                    st.success(
                        f"✅ **{r['filename']}** — "
                        f"{r['pages']} pages, {r['chunks']} chunks"
                    )
                else:
                    st.error(f"❌ **{r['filename']}** — {r.get('error', 'failed')}")

            st.rerun()   # refresh stats

    st.divider()

    # ── 3. Database Stats ──────────────────────────────────────────────────────
    st.subheader("📊 Knowledge Base")

    if st.session_state.collection is not None:
        stats = get_stats(st.session_state.collection)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{stats['total_documents']}</div>
                <div class="stat-lbl">PDFs</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{stats['total_chunks']:,}</div>
                <div class="stat-lbl">Chunks</div>
            </div>""", unsafe_allow_html=True)

        if stats["documents"]:
            st.caption("**Indexed files:**")
            for doc in stats["documents"]:
                st.caption(f"📄 {doc}")
    else:
        st.caption("No data yet. Upload PDFs above.")

    st.divider()

    # ── Clear chat button ──────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =============================================================================
# MAIN AREA — Chat
# =============================================================================

st.title("💬 Chat with your PDFs")
st.caption(f"Model: **gemini-2.5-flash** · Embeddings: **bge-base-en-v1.5** · Vector DB: **ChromaDB**")
st.divider()

# ── Show welcome message if no chat yet ───────────────────────────────────────
if not st.session_state.messages:
    if not st.session_state.ready:
        st.info("👈 Enter your Gemini API key in the sidebar to get started.", icon="🔑")
    elif st.session_state.collection and get_stats(st.session_state.collection)["total_chunks"] == 0:
        st.info("👈 Upload and index some PDFs in the sidebar, then come back to ask questions.", icon="📄")
    else:
        st.info("Type a question below to search your PDF library!", icon="💡")


# ── Display past messages ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(
                f"📚 Sources ({len(msg['sources'])}) · ⚡ {msg.get('latency_ms', 0)}ms",
                expanded=False,
            ):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>📄 {src['filename']}</strong>  ·  Page {src['page']}
                        &nbsp;&nbsp;<span style="color:#22d3ee">↑ {src['vector_score']}% match</span>
                        <br><em style="color:#94a3b8">{src['snippet']}</em>
                    </div>
                    """, unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────────────────────
if question := st.chat_input(
    "Ask a question about your documents…",
    disabled=not st.session_state.ready,
):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching your documents…"):
            try:
                result = ask(
                    question    = question,
                    collection  = st.session_state.collection,
                    api_key     = os.environ.get("GEMINI_API_KEY", ""),
                )
                st.markdown(result["answer"])

                # Show sources inline
                if result["sources"]:
                    with st.expander(
                        f"📚 Sources ({len(result['sources'])}) · ⚡ {result['latency_ms']}ms",
                        expanded=True,
                    ):
                        for src in result["sources"]:
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>📄 {src['filename']}</strong>  ·  Page {src['page']}
                                &nbsp;&nbsp;<span style="color:#22d3ee">↑ {src['vector_score']}% match</span>
                                <br><em style="color:#94a3b8">{src['snippet']}</em>
                            </div>
                            """, unsafe_allow_html=True)


                # Save to history
                st.session_state.messages.append({
                    "role":       "assistant",
                    "content":    result["answer"],
                    "sources":    result["sources"],
                    "latency_ms": result["latency_ms"],
                })

            except Exception as e:
                error_msg = f"⚠️ Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": error_msg,
                    "sources": [],
                })
