"""
DocuMind AI — Command-line interface
Usage: python cli.py [--file data/sample.txt] [--k 3] [--chunk 200]
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


def build_vectorstore(path: str, chunk_size: int, chunk_overlap: int = 20):
    print(f"📂 Loading document: {path}")
    loader = TextLoader(path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(documents)
    print(f"✂️  Split into {len(docs)} chunks")

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
    vectorstore = FAISS.from_documents(docs, embeddings)
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


def ask(query: str, vectorstore, tokenizer, model, k: int) -> tuple[str, list[str]]:
    results = vectorstore.similarity_search(query, k=k)
    chunks = [r.page_content for r in results]
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
    return answer, chunks


def main():
    parser = argparse.ArgumentParser(description="DocuMind AI — CLI RAG chatbot")
    parser.add_argument("--file", default="data/sample.txt", help="Path to .txt document")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--chunk", type=int, default=200, help="Chunk size in characters")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    vectorstore = build_vectorstore(args.file, args.chunk)
    tokenizer, model = load_model()

    print("=" * 60)
    print("💬 DocuMind AI — type your question, or 'exit' to quit")
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

        answer, sources = ask(query, vectorstore, tokenizer, model, args.k)

        print(f"\n🤖 Answer: {answer}")
        print("\n📚 Sources used:")
        for i, chunk in enumerate(sources, 1):
            print(f"  [{i}] {chunk[:120]}{'…' if len(chunk) > 120 else ''}")
        print("-" * 60)


if __name__ == "__main__":
    main()
