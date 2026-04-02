# ================================================================
# backend\ingest.py
# Chunk all PDFs  embed with MiniLM  save FAISS index
# Run: python backend\ingest.py
# ================================================================
import time
import glob
from pathlib import Path

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from backend.config import (
    AGRI_PDF_DIR,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_pdfs(pdf_dir: Path) -> list[dict]:
    """Load all PDFs and return list of {text, source, page}."""
    docs = []
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {pdf_dir}\n")

    for pdf_path in pdfs:
        print(f"  Reading {pdf_path.name}...")
        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if len(text) < 50:   # skip near-empty pages
                    continue
                docs.append({
                    "text"  : text,
                    "source": pdf_path.name,
                    "page"  : page_num,
                })
            print(f"    {len(reader.pages)} pages loaded")
        except Exception as e:
            print(f"    ERROR reading {pdf_path.name}: {e}")

    print(f"\nTotal pages loaded: {len(docs)}")
    return docs


def chunk_docs(docs: list[dict]) -> tuple[list[str], list[dict]]:
    """Split pages into chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size        = CHUNK_SIZE,
        chunk_overlap     = CHUNK_OVERLAP,
        separators        = ["\n\n", "\n", ". ", " ", ""],
        length_function   = len,
    )

    texts     = []
    metadatas = []

    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            if len(chunk.strip()) < 30:   # skip tiny fragments
                continue
            texts.append(chunk)
            metadatas.append({
                "source": doc["source"],
                "page"  : doc["page"],
            })

    print(f"Created {len(texts)} chunks from {len(docs)} pages")
    return texts, metadatas


def build_faiss(texts: list[str], metadatas: list[dict]) -> None:
    """Embed all chunks and save FAISS index to disk."""
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads ~120MB — takes 2-3 minutes)\n")

    embeddings = HuggingFaceEmbeddings(
        model_name     = EMBEDDING_MODEL,
        model_kwargs   = {"device": "cpu"},
        encode_kwargs  = {"normalize_embeddings": True},
    )

    print(f"Embedding {len(texts)} chunks...")
    print("This takes 5-8 minutes on CPU — do not interrupt\n")

    start = time.time()
    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    duration = round(time.time() - start, 1)

    db.save_local(str(VECTORSTORE_DIR))
    print(f"\nFAISS index saved to: {VECTORSTORE_DIR}")
    print(f"Total vectors stored: {db.index.ntotal}")
    print(f"Time taken: {duration}s")


if __name__ == "__main__":
    print("=" * 55)
    print("INGEST PIPELINE — Crop Disease Detector")
    print("=" * 55 + "\n")

    docs               = load_pdfs(AGRI_PDF_DIR)
    texts, metadatas   = chunk_docs(docs)

    if len(texts) < 100:
        print("\nWARNING: fewer than 100 chunks — check your PDFs")
    else:
        print(f"\nChunk count OK: {len(texts)} chunks")

    build_faiss(texts, metadatas)

    print("\n" + "=" * 55)
    print("INGEST COMPLETE")
    print(f"Run: python backend/test_faiss.py to verify")
    print("=" * 55)
