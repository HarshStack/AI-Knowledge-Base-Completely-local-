import os
import pickle
import numpy as np
import faiss
import nltk
import pdfplumber
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pymongo import UpdateOne
from db import get_chunks_collection, ensure_indexes, now_utc

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_pdfs(folder="data"):
    """
    Recursively load all PDFs from the given folder and its subfolders.
    Returns a list of page-level dicts with text, table_chunks, source, and page.
    """
    docs = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)

                # Use a relative source path so filenames stay meaningful
                rel_path = os.path.relpath(path, folder)

                try:
                    with pdfplumber.open(path) as pdf:
                        for page_num, page in enumerate(pdf.pages):
                            text = page.extract_text() or ""
                            tables = page.extract_tables()

                            table_chunks = []
                            for table in tables:
                                if not table:
                                    continue
                                headers = table[0]
                                for row in table[1:]:
                                    row_data = []
                                    for h, cell in zip(headers, row):
                                        if h and cell:
                                            row_data.append(
                                                f"{h.strip()}: {cell.strip()}"
                                            )
                                    if row_data:
                                        table_chunks.append(" | ".join(row_data))

                            docs.append(
                                {
                                    "text": text,
                                    "table_chunks": table_chunks,
                                    "source": rel_path,
                                    "page": page_num,
                                }
                            )
                except Exception as e:
                    print(f"⚠ Failed to load {path}: {e}")

    return docs


def split_text(text, max_chars=800):
    """
    Sentence-aware chunking.
    Groups sentences until max_chars is reached.
    Prevents mid-sentence cuts.
    """
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def build_index(data_folder="data"):
    """
    Build FAISS + BM25 index from all PDFs found under data_folder (recursively).
    Also upserts chunks into MongoDB if available.
    """
    os.makedirs("faiss_index", exist_ok=True)

    raw_docs = load_pdfs(data_folder)
    if not raw_docs:
        print(f"⚠ No PDF pages found under '{data_folder}'. Aborting.")
        return

    all_chunks = []
    metadata = []

    for doc in raw_docs:
        # ── Text chunking ──────────────────────────────────────────────────
        parts = split_text(doc["text"], CHUNK_SIZE)
        for idx, p in enumerate(parts):
            all_chunks.append(p)
            metadata.append(
                {
                    "source": doc["source"],
                    "page": doc["page"],
                    "chunk_index": idx,
                    "chunk_type": "text",
                }
            )

        # ── Table chunking ─────────────────────────────────────────────────
        for t_idx, row in enumerate(doc.get("table_chunks", [])):
            all_chunks.append(row)
            metadata.append(
                {
                    "source": doc["source"],
                    "page": doc["page"],
                    "chunk_index": 10000 + t_idx,
                    "chunk_type": "table",
                }
            )

    print(f"PDF pages loaded   : {len(raw_docs)}")
    print(f"Total chunks       : {len(all_chunks)}")

    # ── Embeddings ─────────────────────────────────────────────────────────
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embedder.encode(all_chunks, convert_to_numpy=True).astype("float32")

    # ── MongoDB (optional) ──────────────────────────────────────────────────
    col = get_chunks_collection()
    if col is not None:
        ensure_indexes()
        ops = []
        for text, meta, vec in zip(all_chunks, metadata, vectors):
            selector = {
                "source": meta["source"],
                "page": meta["page"],
                "chunk_index": meta["chunk_index"],
            }
            doc = {
                **selector,
                "text": text,
                "embedding": vec.tolist(),
                "created_at": now_utc(),
            }
            ops.append(UpdateOne(selector, {"$set": doc}, upsert=True))
        col.bulk_write(ops)
        print(f"✔ Stored {len(ops)} chunks in MongoDB")
    else:
        print("⚠ MongoDB not available. Skipping storage.")

    # ── FAISS ───────────────────────────────────────────────────────────────
    dim = vectors.shape[1]
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, "faiss_index/index.faiss")

    # ── BM25 ────────────────────────────────────────────────────────────────
    tokenized = [t.split() for t in all_chunks]
    bm25 = BM25Okapi(tokenized)

    with open("faiss_index/metadata.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "meta": metadata, "bm25": bm25}, f)

    print("✔ FAISS + BM25 index built successfully!")


if __name__ == "__main__":
    build_index()
