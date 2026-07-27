"""
DocuMind AI — Command-line interface
Usage:
  python cli.py --file data/stories.txt
  python cli.py --all   # Search across all documents in data/
"""

import argparse
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vectorstore(data_path: str, chunk_size: int = 500, is_all: bool = False, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_docs = []

    if is_all or os.path.isdir(data_path):
        folder = data_path if os.path.isdir(data_path) else "data"
        print(f"📂 Loading all documents from directory: '{folder}'")
        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        for fname in files:
            fpath = os.path.join(folder, fname)
            loader = TextLoader(fpath, encoding="utf-8")
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = fname
            all_docs.extend(splitter.split_documents(docs))
    else:
        print(f"📂 Loading document: '{data_path}'")
        fname = os.path.basename(data_path)
        loader = TextLoader(data_path, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = fname
        all_docs.extend(splitter.split_documents(docs))

    print(f"✂️  Split into {len(all_docs)} chunks")
    print("🔍 Building vector index (this may take a moment)…")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    except Exception:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu", "local_files_only": True}
        )
        
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    print("✅ Vector index ready\n")
    return vectorstore


def load_model():
    print("🤖 Loading Flan-T5 model…")
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", local_files_only=True)
    print("✅ Model ready\n")
    return tokenizer, model


def ask(query: str, vectorstore, tokenizer, model, k: int, chat_history: list) -> tuple[str, list]:
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    retrieved_items = []
    chunks = []
    for doc, dist in results:
        score_pct = round(max(0.0, 1.0 - (dist / 2.0)) * 100, 1)
        src_name = os.path.basename(doc.metadata.get("source", "doc"))
        retrieved_items.append({"text": doc.page_content, "score": score_pct, "source": src_name})
        chunks.append(doc.page_content)

    context = "\n\n".join(chunks)

    # Include recent chat history
    history_str = ""
    if chat_history:
        recent = chat_history[-4:]  # Last 2 turns
        lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent]
        history_str = "\nPrevious Conversation:\n" + "\n".join(lines) + "\n"

    prompt = f"""Answer the question based ONLY on the context below. Be concise and factual.

Context:
{context}
{history_str}
Question: {query}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer, retrieved_items


def main():
    parser = argparse.ArgumentParser(description="DocuMind AI — CLI RAG Chatbot")
    parser.add_argument("--file", default="data/sample.txt", help="Path to .txt document")
    parser.add_argument("--all", action="store_true", help="Search across ALL documents in data/")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--chunk", type=int, default=500, help="Chunk size in characters")
    args = parser.parse_args()

    if not args.all and not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    vectorstore = build_vectorstore(args.file, args.chunk, is_all=args.all)
    tokenizer, model = load_model()

    chat_history = []

    print("=" * 60)
    print("💬 DocuMind AI CLI — Type your question, or 'exit' to quit")
    print("   Multi-turn chat memory is active!")
    print("=" * 60)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            print("👋 Goodbye!")
            break

        answer, sources = ask(query, vectorstore, tokenizer, model, args.k, chat_history)

        print(f"\n🤖 Answer: {answer}")
        print("\n📚 Sources used:")
        for i, src in enumerate(sources, 1):
            score_formatted = round(float(src['score']), 1)
            print(f"  [{i}] [{src['source']}] (Match: {score_formatted}%) -> {src['text'][:100]}…")
        print("-" * 60)

        # Store in conversation memory
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
