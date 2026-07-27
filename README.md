# 📄 DocuMind AI

A local RAG (Retrieval-Augmented Generation) chatbot that lets you ask questions about any plain-text document. Runs fully offline using **Flan-T5** + **FAISS** + **sentence-transformers**.

---

## Project Structure

```
documind/
├── app.py              # Streamlit web UI
├── cli.py              # Command-line interface
├── requirements.txt    # Python dependencies
├── data/
│   └── sample.txt      # Sample knowledge base (RAG concepts)
└── README.md
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run will download ~500 MB of model weights (Flan-T5 + MiniLM). Cached after that.

---

## Running

### Option A — Streamlit Web UI

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**Features:**
- Document picker (any `.txt` file placed in `data/`)
- Adjustable chunk size and retrieval count via sidebar
- Chat history with source chunk viewer

### Option B — Command-line Interface

```bash
python cli.py
```

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--file` | `data/sample.txt` | Path to your `.txt` document |
| `--k` | `3` | Number of chunks to retrieve |
| `--chunk` | `200` | Chunk size in characters |

Example with a custom document:
```bash
python cli.py --file data/my_notes.txt --k 4 --chunk 300
```

---

## Using Your Own Document

1. Place any `.txt` file inside the `data/` folder.
2. **Streamlit:** select it from the sidebar dropdown.
3. **CLI:** pass it with `--file data/yourfile.txt`.

---

## How It Works

```
Your question
     │
     ▼
[Embedding model]  ←  sentence-transformers/all-MiniLM-L6-v2
     │
     ▼
[FAISS vector search]  →  top-k relevant chunks from document
     │
     ▼
[Prompt assembly]  →  "Answer only from context: {chunks} Question: {query}"
     │
     ▼
[Flan-T5 generation]  →  grounded answer
```

---

## Models Used

| Component | Model | Size |
|-----------|-------|------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB |
| Generation | `google/flan-t5-base` | ~250 MB |

Both run on CPU — no GPU required.
