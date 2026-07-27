import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="centered",
)

# ── Custom CSS Design System ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
    }

    .main-title {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .badge-high {
        background: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-medium {
        background: rgba(245, 158, 11, 0.18);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-low {
        background: rgba(99, 102, 241, 0.18);
        color: #818cf8;
        border: 1px solid rgba(129, 140, 248, 0.3);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-file {
        background: rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 6px;
    }

    .stExpander {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        margin-top: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 DocuMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions from your document — powered by Hybrid RAG + Flan-T5</div>', unsafe_allow_html=True)

# ── Helpers: Document Text Extractors ──────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception:
        return file_bytes.decode("latin1", errors="ignore")


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    except Exception:
        return file_bytes.decode("utf-8", errors="ignore")


# ── Sidebar: file uploader & settings ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings & Document")

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # File Uploader (.pdf, .txt, .docx, .md)
    uploaded_file = st.file_uploader("📥 Upload Document", type=["pdf", "txt", "docx", "md"])
    if uploaded_file is not None:
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()

        if filename.endswith(".pdf"):
            text_content = extract_text_from_pdf(file_bytes)
            txt_filename = filename[:-4] + ".txt"
            save_path = os.path.join(data_dir, txt_filename)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            st.success(f"✅ Saved & parsed PDF: `{txt_filename}`")
        elif filename.endswith(".docx"):
            text_content = extract_text_from_docx(file_bytes)
            txt_filename = filename[:-5] + ".txt"
            save_path = os.path.join(data_dir, txt_filename)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            st.success(f"✅ Saved & parsed Word Doc: `{txt_filename}`")
        else:
            save_path = os.path.join(data_dir, filename)
            with open(save_path, "wb") as f:
                f.write(file_bytes)
            st.success(f"✅ Saved file: `{filename}`")

    # Document Picker
    all_files = [f for f in os.listdir(data_dir) if f.endswith((".txt", ".md"))]
    if not all_files:
        st.error("No documents found in /data. Please upload a file above.")
        st.stop()

    doc_options = ["🌐 All Documents (Combined)"] + sorted(all_files)
    selected_file = st.selectbox("📂 Choose a document", doc_options)

    st.markdown("---")
    st.subheader("🔍 Retrieval Settings")
    search_mode = st.radio("Search Mode", ["Hybrid (FAISS + Keyword RRF)", "Dense Only (FAISS)"])
    chunk_size = st.slider("Chunk size (characters)", 100, 1000, 500, step=50)
    top_k = st.slider("Chunks retrieved (k)", 1, 6, 3)

    st.markdown("---")
    st.markdown("**Model:** `google/flan-t5-base`")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2`")

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Export Transcript Button
    if "messages" in st.session_state and st.session_state.messages:
        report_md = "# DocuMind AI — Q&A Chat Transcript\n\n"
        for m in st.session_state.messages:
            role_title = "👤 User" if m["role"] == "user" else "🤖 Assistant"
            report_md += f"### {role_title}\n{m['content']}\n\n"
        st.download_button(
            label="📥 Export Chat Report (.md)",
            data=report_md,
            file_name="documind_chat_report.md",
            mime="text/markdown",
            use_container_width=True
        )

# ── Auto-reset chat on document change ─────────────────────────────────────────
if "current_file" not in st.session_state:
    st.session_state.current_file = selected_file
elif st.session_state.current_file != selected_file:
    st.session_state.messages = []
    st.session_state.current_file = selected_file

# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Loading embeddings model…")
def load_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    except Exception:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu", "local_files_only": True}
        )


@st.cache_resource(show_spinner="🔍 Building vector index…")
def load_vectorstore(data_folder: str, file_choice: str, chunk: int):
    embeddings = load_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=100)
    all_docs = []

    if file_choice == "🌐 All Documents (Combined)":
        files_to_load = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    else:
        files_to_load = [file_choice]

    for fname in files_to_load:
        fpath = os.path.join(data_folder, fname)
        if os.path.exists(fpath):
            loader = TextLoader(fpath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = fname
            split = splitter.split_documents(docs)
            all_docs.extend(split)

    vs = FAISS.from_documents(all_docs, embeddings)
    return vs, all_docs


@st.cache_resource(show_spinner="🤖 Loading language model…")
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", local_files_only=True)
    return tokenizer, model


vectorstore, loaded_chunks = load_vectorstore(data_dir, selected_file, chunk_size)
tokenizer, model = load_model()

# ── Hybrid Retrieval (FAISS + Keyword RRF) ────────────────────────────────────
def get_retrieved_chunks(query: str, k: int, mode: str) -> list[dict]:
    if mode == "Dense Only (FAISS)":
        results = vectorstore.similarity_search_with_score(query, k=k)
        items = []
        for doc, dist in results:
            score_pct = round(max(0.0, 1.0 - (dist / 2.0)) * 100, 1)
            src_name = os.path.basename(doc.metadata.get("source", "doc"))
            items.append({"text": doc.page_content, "score": score_pct, "source": src_name})
        return items

    # Hybrid Search with Reciprocal Rank Fusion (RRF)
    candidate_docs = vectorstore.similarity_search_with_score(query, k=min(k * 3, 15))
    chunks = [doc.page_content for doc, _ in candidate_docs]
    sources = [os.path.basename(doc.metadata.get("source", "doc")) for doc, _ in candidate_docs]
    keywords = [w.lower() for w in query.split() if len(w) > 2]

    def keyword_score(text: str) -> float:
        text_lower = text.lower()
        return sum(text_lower.count(kw) for kw in keywords)

    sorted_by_kw = sorted(range(len(chunks)), key=lambda i: keyword_score(chunks[i]), reverse=True)
    kw_ranks = {idx: rank for rank, idx in enumerate(sorted_by_kw)}

    rrf_scores = []
    for dense_rank, chunk in enumerate(chunks):
        kw_rank = kw_ranks[dense_rank]
        rrf_score = (1.0 / (60 + dense_rank)) + (1.0 / (60 + kw_rank))
        rrf_scores.append((rrf_score, chunk, sources[dense_rank]))

    rrf_scores.sort(key=lambda x: x[0], reverse=True)
    items = []
    for score, chunk, src_name in rrf_scores[:k]:
        score_pct = round(min(100.0, (score / 0.03333) * 100), 1)
        items.append({"text": chunk, "score": score_pct, "source": src_name})
    return items

def render_sources(sources):
    with st.expander("📚 Source chunks used"):
        for i, src in enumerate(sources, 1):
            if isinstance(src, dict):
                src_file = src.get("source", "Document")
                score = src.get("score", 0)
                if score >= 80:
                    badge_class = "badge-high"
                elif score >= 50:
                    badge_class = "badge-medium"
                else:
                    badge_class = "badge-low"

                st.markdown(
                    f"**Chunk {i}** <span class='badge-file'>📄 {src_file}</span> "
                    f"<span class='{badge_class}'>🎯 {score}% Match</span>\n\n"
                    f"> {src['text']}",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"**Chunk {i}:**\n> {src}")

# ── Main Layout Tabs ──────────────────────────────────────────────────────────
tab_chat, tab_viz = st.tabs(["💬 Chatbot", "📊 Document Inspection & Vector Visualizer"])

with tab_chat:
    # Render previous messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                render_sources(msg["sources"])

    # RAG query function
    def ask(query: str, k: int) -> tuple[str, list[dict]]:
        sources = get_retrieved_chunks(query, k, search_mode)
        chunks = [s["text"] if isinstance(s, dict) else s for s in sources]
        context = "\n\n".join(chunks)

        # Multi-turn conversation memory from recent messages
        history_str = ""
        if st.session_state.messages:
            recent = st.session_state.messages[-4:]  # Last 2 conversation turns
            history_lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent]
            history_str = "\nPrevious Conversation:\n" + "\n".join(history_lines) + "\n"

        prompt = f"""Answer the question based ONLY on the context below. Be concise and factual.

Context:
{context}
{history_str}
Question: {query}
Answer:"""

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=256)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return answer, sources

    # Chat input
    if user_input := st.chat_input("Ask a question about your document…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer, sources = ask(user_input, top_k)

            st.markdown(answer)
            render_sources(sources)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

    if not st.session_state.messages:
        st.info(
            "👆 Type a question in the chat box below. "
            f"Currently loaded: **{selected_file}** | Mode: **{search_mode}**"
        )

with tab_viz:
    st.subheader("📊 Index & Vector Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chunks Indexing", len(loaded_chunks))
    col2.metric("Chunk Size Limit", f"{chunk_size} chars")
    col3.metric("Embedding Dimension", "384 (MiniLM)")
    col4.metric("Active Selection", "All Docs" if selected_file.startswith("🌐") else "1 Document")

    st.markdown("---")
    st.subheader("🔍 Text Chunk Explorer")

    doc_sources = sorted(list(set(doc.metadata.get("source", "doc") for doc in loaded_chunks)))
    filter_doc = st.selectbox("Filter chunks by source document:", ["All Documents"] + doc_sources)

    filtered_chunks = [
        doc for doc in loaded_chunks
        if filter_doc == "All Documents" or doc.metadata.get("source", "") == filter_doc
    ]

    st.caption(f"Showing **{len(filtered_chunks)}** chunk(s)")

    for idx, chunk_doc in enumerate(filtered_chunks[:50], 1):
        src_file = chunk_doc.metadata.get("source", "doc")
        char_len = len(chunk_doc.page_content)
        with st.expander(f"Chunk #{idx} | 📄 {src_file} ({char_len} characters)"):
            st.code(chunk_doc.page_content, language="markdown")
