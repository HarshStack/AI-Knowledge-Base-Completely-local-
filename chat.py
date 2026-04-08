
import os
import pickle
import faiss
import re
import numpy as np
from dotenv import load_dotenv

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from sentence_transformers import CrossEncoder

load_dotenv()

TOP_K = int(os.getenv("TOP_K", 3))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


class Streamer(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)


def load_index():
    index = faiss.read_index("faiss_index/index.faiss")
    with open("faiss_index/metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    return index, meta["chunks"], meta["meta"], meta["bm25"]


def hybrid_search(query, index, texts, metadatas, bm25, embed, k=15):
    k_dense = 12
    k_sparse = 12

    # ── Sparse Retrieval (BM25) ────────────────────────────────────────────
    tokens = re.findall(r"\w+", query.lower())
    bm_scores = np.array(bm25.get_scores(tokens))
    bm_top_ids = np.argsort(bm_scores)[::-1][:k_sparse]
    bm_selected = bm_scores[bm_top_ids]

    if bm_selected.max() - bm_selected.min() > 0:
        bm_norm = (bm_selected - bm_selected.min()) / (
            bm_selected.max() - bm_selected.min()
        )
    else:
        bm_norm = np.zeros_like(bm_selected)

    # ── Dense Retrieval (FAISS) ────────────────────────────────────────────
    q_vec = embed.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)

    dense_scores, dense_ids = index.search(q_vec, k_dense)
    dense_scores = dense_scores[0]
    dense_ids = dense_ids[0]

    if dense_scores.max() - dense_scores.min() > 0:
        dense_norm = (dense_scores - dense_scores.min()) / (
            dense_scores.max() - dense_scores.min()
        )
    else:
        dense_norm = np.zeros_like(dense_scores)

    # ── Fusion ─────────────────────────────────────────────────────────────
    candidate_ids = list(set(bm_top_ids.tolist() + dense_ids.tolist()))

    alpha = 0.6  # dense weight
    beta = 0.4   # sparse weight
    is_numeric_query = bool(re.search(r"\d", query))

    final_scores = []
    for cid in candidate_ids:
        dense_score = 0.0
        sparse_score = 0.0

        if cid in dense_ids.tolist():
            idx = dense_ids.tolist().index(cid)
            dense_score = dense_norm[idx]

        if cid in bm_top_ids.tolist():
            idx = bm_top_ids.tolist().index(cid)
            sparse_score = bm_norm[idx]

        final = alpha * dense_score + beta * sparse_score

        if metadatas[cid].get("chunk_type") == "table":
            final += 0.15
        if is_numeric_query and metadatas[cid].get("chunk_type") == "table":
            final += 0.15

        final_scores.append((cid, final))

    final_scores.sort(key=lambda x: x[1], reverse=True)
    candidate_ids = [cid for cid, _ in final_scores[:15]]

    return [{"text": texts[i], "meta": metadatas[i]} for i in candidate_ids]


def expand_chunks(docs, texts, metadatas, window=1):
    """Expands retrieved chunks to include neighbouring chunks."""
    expanded_ids = set()
    for doc in docs:
        idx = texts.index(doc["text"])
        for i in range(idx - window, idx + window + 1):
            if 0 <= i < len(texts):
                expanded_ids.add(i)
    return [{"text": texts[i], "meta": metadatas[i]} for i in expanded_ids]


def chat_loop():
    index, texts, metadatas, bm25 = load_index()
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        streaming=True,
        callbacks=[Streamer()],
        keep_alive="5m",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer ONLY using the provided context."),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    print("\n Local RAG Ready \n")
    while True:
        q = input("\nQ: ").strip()
        if q.lower() == "exit":
            break

        docs = hybrid_search(q, index, texts, metadatas, bm25, embed)
        docs = expand_chunks(docs, texts, metadatas, window=1)

        pairs = [(q, d["text"]) for d in docs]
        scores = reranker.predict(pairs)

        for i, s in enumerate(scores):
            docs[i]["rerank_score"] = float(s)

        docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:TOP_K]

        context = "\n\n".join(
            f"[{d['meta']['source']}] {d['text']}" for d in docs
        )

        print("\n--- Answer ---\n")
        answer = chain.invoke({"question": q, "context": context})

        verification_prompt = f"""
You are a verification assistant.
Check if the following answer is fully supported by the context.
If fully supported, respond with: VERIFIED
If not fully supported, respond with: NOT VERIFIED

Answer:
{answer}

Context:
{context}
"""
        verification = llm.invoke(verification_prompt)

        print("\n--- Answer ---\n")
        print(answer)
        print("\n--- Verification ---\n")
        print(verification)


if __name__ == "__main__":
    chat_loop()
