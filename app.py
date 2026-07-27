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

st.title("📄 DocuMind AI")
st.caption("Ask questions from your document — powered by Hybrid RAG + Flan-T5")

# ── Helper: PDF Text Extractor ────────────────────────────────────────────────
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
    except ImportError:
        st.warning("⚠️ `pypdf` is not installed. Please run `pip install pypdf` to parse PDFs.")
        return file_bytes.decode("latin1", errors="ignore")

# ── Sidebar: file uploader & settings ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings & Document")

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # File Uploader
    uploaded_file = st.file_uploader("📥 Upload a PDF or TXT file", type=["pdf", "txt"])
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
        else:
            save_path = os.path.join(data_dir, filename)
            with open(save_path, "wb") as f:
                f.write(file_bytes)
            st.success(f"✅ Saved file: `{filename}`")

    # Document Picker
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if not all_files:
        st.error("No documents found in /data. Please upload a file above.")
        st.stop()

    selected_file = st.selectbox("📂 Choose a document", all_files)
    doc_path = os.path.join(data_dir, selected_file)

    st.markdown("---")
    st.subheader("🔍 Retrieval Settings")
    search_mode = st.radio("Search Mode", ["Hybrid (FAISS + Keyword RRF)", "Dense Only (FAISS)"])
    chunk_size = st.slider("Chunk size (tokens)", 50, 500, 200, step=50)
    top_k = st.slider("Chunks retrieved (k)", 1, 6, 3)

    st.markdown("---")
    st.markdown("**Model:** `google/flan-t5-base`")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2`")

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

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
def load_vectorstore(path: str, chunk: int):
    loader = TextLoader(path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=20)
    docs = splitter.split_documents(documents)

    embeddings = load_embeddings()
    return FAISS.from_documents(docs, embeddings)


@st.cache_resource(show_spinner="🤖 Loading language model…")
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", local_files_only=True)
    return tokenizer, model


vectorstore = load_vectorstore(doc_path, chunk_size)
tokenizer, model = load_model()

# ── Hybrid Retrieval (FAISS + Keyword RRF) ────────────────────────────────────
def get_retrieved_chunks(query: str, k: int, mode: str) -> list[dict]:
    if mode == "Dense Only (FAISS)":
        results = vectorstore.similarity_search_with_score(query, k=k)
        items = []
        for doc, dist in results:
            score_pct = round(max(0.0, 1.0 - (dist / 2.0)) * 100, 1)
            items.append({"text": doc.page_content, "score": score_pct})
        return items

    # Hybrid Search with Reciprocal Rank Fusion (RRF)
    candidate_docs = vectorstore.similarity_search_with_score(query, k=min(k * 3, 15))
    chunks = [doc.page_content for doc, _ in candidate_docs]
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
        rrf_scores.append((rrf_score, chunk))

    rrf_scores.sort(key=lambda x: x[0], reverse=True)
    items = []
    for score, chunk in rrf_scores[:k]:
        score_pct = round(min(100.0, (score / 0.03333) * 100), 1)
        items.append({"text": chunk, "score": score_pct})
    return items

def render_sources(sources):
    with st.expander("📚 Source chunks used"):
        for i, src in enumerate(sources, 1):
            if isinstance(src, dict):
                st.markdown(f"**Chunk {i}** *(Relevance Score: `{src['score']}%`)*:\n> {src['text']}")
            else:
                st.markdown(f"**Chunk {i}:**\n> {src}")

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            render_sources(msg["sources"])

# ── RAG query function ────────────────────────────────────────────────────────
def ask(query: str, k: int) -> tuple[str, list[dict]]:
    sources = get_retrieved_chunks(query, k, search_mode)
    chunks = [s["text"] if isinstance(s, dict) else s for s in sources]
    context = "\n\n".join(chunks)

    prompt = f"""You are a helpful RAG assistant. Answer the user's question based ONLY on the context provided.
If the context does not contain the answer or does not mention the person/topic, clearly state: "The provided document does not contain information about '{query}'."

Context:
{context}

Question: {query}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer, sources

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a question about your document…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
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

# ── Empty state hint ──────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.info(
        "👆 Type a question in the chat box below. "
        f"Currently loaded: **{selected_file}** | Mode: **{search_mode}**"
    )
