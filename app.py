
import os
import time
import zipfile
import tempfile
import shutil
import pickle

import streamlit as st
from dotenv import load_dotenv

# PDF pipeline
from ingest_local import build_index as build_pdf_index
from chat_local import load_index, hybrid_search

# Excel pipeline
from ingest_excel import build_excel_index, get_schema, excel_index_exists
from excel_qa import ask_excel

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
DATA_DIR = "data"

# ──────────────────────────────────────────────────────────────────────────────
# Streaming callback
# ──────────────────────────────────────────────────────────────────────────────

class StreamlitStreamer(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.placeholder.markdown(self.text + "▌")


# ──────────────────────────────────────────────────────────────────────────────
# Cached model loaders
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ──────────────────────────────────────────────────────────────────────────────
# File-saving helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_single_files(uploaded_files: list, allowed_exts: tuple) -> list[str]:
    """Save individually uploaded files into DATA_DIR. Returns saved filenames."""
    os.makedirs(DATA_DIR, exist_ok=True)
    saved = []
    for uf in uploaded_files:
        if any(uf.name.lower().endswith(e) for e in allowed_exts):
            dest = os.path.join(DATA_DIR, uf.name)
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved.append(uf.name)
    return saved


def save_folder_zip(zip_file, allowed_exts: tuple) -> tuple[int, list[str]]:
    """
    Extract all matching files from a ZIP archive into DATA_DIR,
    preserving sub-folder structure.
    Returns (count, list_of_relative_paths).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    saved_paths = []

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                bname = os.path.basename(member.filename)
                if bname.startswith("._") or "__MACOSX" in member.filename:
                    continue
                if not any(member.filename.lower().endswith(e) for e in allowed_exts):
                    continue

                parts = [p for p in member.filename.split("/") if p]
                rel_parts = parts[1:] if len(parts) > 1 else parts
                rel_path = os.path.join(*rel_parts)

                dest = os.path.join(DATA_DIR, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)

                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                saved_paths.append(rel_path)

    return len(saved_paths), saved_paths


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Local RAG + Excel Q&A", layout="wide")
st.title("🔍 Local RAG + Excel Q&A")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – Upload zone (tabs: PDF | Excel)
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Upload & Index")

    pdf_tab, excel_tab = st.tabs(["PDFs", "Excel / CSV"])

    # ── PDF tab ───────────────────────────────────────────────────────────
    with pdf_tab:
        pdf_mode = st.radio(
            "Upload mode",
            ["Single / multiple PDFs", "Folder (ZIP archive)"],
            key="pdf_mode",
        )

        uploaded_pdfs = None
        uploaded_pdf_zip = None

        if pdf_mode == "Single / multiple PDFs":
            uploaded_pdfs = st.file_uploader(
                "Choose PDF file(s)",
                type=["pdf"],
                accept_multiple_files=True,
                key="pdf_uploader",
            )
        else:
            uploaded_pdf_zip = st.file_uploader(
                "ZIP archive of PDFs",
                type=["zip"],
                key="pdf_zip_uploader",
            )

        if st.button("Build PDF Index", use_container_width=True):
            files_ready = False

            if pdf_mode == "Single / multiple PDFs" and uploaded_pdfs:
                saved = save_single_files(uploaded_pdfs, (".pdf",))
                st.success(f"Saved {len(saved)} PDF(s).")
                files_ready = True
            elif pdf_mode == "Folder (ZIP archive)" and uploaded_pdf_zip:
                count, paths = save_folder_zip(uploaded_pdf_zip, (".pdf",))
                if count == 0:
                    st.error("No PDFs found in ZIP.")
                else:
                    st.success(f"Extracted {count} PDF(s).")
                    files_ready = True
            else:
                existing = (
                    [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
                    if os.path.isdir(DATA_DIR) else []
                )
                if existing:
                    st.info(f"Building from {len(existing)} existing PDF(s).")
                    files_ready = True
                else:
                    st.warning("Upload PDFs first.")

            if files_ready:
                with st.spinner("Building PDF index…"):
                    try:
                        build_pdf_index(DATA_DIR)
                        st.success("PDF index ready!")
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        # Show indexed PDF sources
        if os.path.exists("faiss_index/metadata.pkl"):
            with open("faiss_index/metadata.pkl", "rb") as f:
                meta = pickle.load(f)
            sources = sorted({m["source"] for m in meta["meta"]})
            with st.expander(f"Indexed PDFs ({len(sources)})"):
                for s in sources:
                    st.markdown(f"• `{s}`")

    # ── Excel tab ─────────────────────────────────────────────────────────
    with excel_tab:
        excel_mode = st.radio(
            "Upload mode",
            ["Single / multiple files", "Folder (ZIP archive)"],
            key="excel_mode",
        )

        uploaded_excels = None
        uploaded_excel_zip = None
        EXCEL_EXTS = (".xlsx", ".xls", ".csv")

        if excel_mode == "Single / multiple files":
            uploaded_excels = st.file_uploader(
                "Choose Excel / CSV file(s)",
                type=["xlsx", "xls", ".csv"],
                accept_multiple_files=True,
                key="excel_uploader",
            )
        else:
            uploaded_excel_zip = st.file_uploader(
                "ZIP archive containing Excel/CSV files",
                type=["zip"],
                key="excel_zip_uploader",
            )

        if st.button("⚙️ Build Excel Index", use_container_width=True):
            files_ready = False

            if excel_mode == "Single / multiple files" and uploaded_excels:
                saved = save_single_files(uploaded_excels, EXCEL_EXTS)
                st.success(f"Saved {len(saved)} file(s): {', '.join(saved)}")
                files_ready = True
            elif excel_mode == "Folder (ZIP archive)" and uploaded_excel_zip:
                count, paths = save_folder_zip(uploaded_excel_zip, EXCEL_EXTS)
                if count == 0:
                    st.error("No Excel/CSV files found in ZIP.")
                else:
                    st.success(f"Extracted {count} file(s).")
                    files_ready = True
            else:
                existing = (
                    [
                        f for f in os.listdir(DATA_DIR)
                        if any(f.lower().endswith(e) for e in EXCEL_EXTS)
                    ]
                    if os.path.isdir(DATA_DIR) else []
                )
                if existing:
                    st.info(f"Building from {len(existing)} existing file(s).")
                    files_ready = True
                else:
                    st.warning("Upload Excel or CSV files first.")

            if files_ready:
                with st.spinner("Building Excel index…"):
                    try:
                        n = build_excel_index(DATA_DIR)
                        if n:
                            st.success(f"Excel index ready! ({n} table(s))")
                        else:
                            st.warning("No tables were indexed.")
                    except Exception as e:
                        st.error(f"Failed: {e}")

        # Show indexed Excel tables
        if excel_index_exists():
            schema = get_schema()
            with st.expander(f"Indexed tables ({len(schema)})"):
                for entry in schema:
                    cols = ", ".join(c["name"] for c in entry["columns"])
                    st.markdown(
                        f"**`{entry['table_name']}`** "
                        f"({entry['row_count']} rows)  \n"
                        f"*{entry['source']} / {entry['sheet']}*  \n"
                        f"Columns: `{cols}`"
                    )
                    st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Main Q&A area – two tabs: PDF RAG | Excel Q&A
# ──────────────────────────────────────────────────────────────────────────────

pdf_qa_tab, excel_qa_tab = st.tabs(["Ask your PDFs", "Ask your Excel data"])


# ── PDF Q&A ───────────────────────────────────────────────────────────────────
with pdf_qa_tab:
    if not os.path.exists("faiss_index/index.faiss"):
        st.info("Upload PDFs and click **Build PDF Index** to get started.")
    else:
        pdf_question = st.text_input(
            "Ask a question about your PDF documents:",
            key="pdf_question",
        )

        if pdf_question:
            embed = load_embedder()
            reranker = load_reranker()

            with st.chat_message("user"):
                st.markdown(pdf_question)

            with st.chat_message("assistant"):
                progress = st.progress(0)
                status = st.empty()

                t0 = time.time()
                status.markdown("Retrieving chunks…")
                progress.progress(30)

                index, texts, metadatas, bm25 = load_index()
                docs = hybrid_search(
                    pdf_question, index, texts, metadatas, bm25, embed, k=10
                )

                progress.progress(60)
                status.markdown("Re-ranking…")

                pairs = [(pdf_question, d["text"]) for d in docs]
                scores = reranker.predict(pairs)
                for i, s in enumerate(scores):
                    docs[i]["score"] = float(s)
                docs = sorted(docs, key=lambda x: x["score"], reverse=True)[:3]
                context = "\n\n".join(d["text"] for d in docs)

                progress.progress(100)
                retrieval_time = round(time.time() - t0, 2)
                progress.empty()
                status.empty()

                placeholder = st.empty()
                streamer = StreamlitStreamer(placeholder)

                llm = ChatOllama(
                    model=OLLAMA_MODEL, streaming=True, callbacks=[streamer]
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer only using the context provided."),
                    ("human", "Question: {question}\n\nContext:\n{context}"),
                ])
                chain = prompt | llm | StrOutputParser()

                t1 = time.time()
                chain.invoke({"question": pdf_question, "context": context})
                llm_time = round(time.time() - t1, 2)
                placeholder.markdown(streamer.text)

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Retrieval", f"{retrieval_time}s")
                c2.metric("LLM", f"{llm_time}s")
                c3.metric("Total", f"{retrieval_time + llm_time}s")

                with st.expander("Retrieved Context"):
                    for d in docs:
                        st.markdown(
                            f"**Source:** `{d['meta']['source']}` — "
                            f"Page {d['meta']['page']} | Score: `{d['score']:.3f}`"
                        )
                        st.write(d["text"][:500])
                        st.divider()


# ── Excel Q&A ─────────────────────────────────────────────────────────────────
with excel_qa_tab:
    if not excel_index_exists():
        st.info(
            "Upload Excel/CSV files and click **Build Excel Index** to get started."
        )
    else:
        schema = get_schema()

        with st.expander("📋 Available tables (schema reference)", expanded=False):
            for entry in schema:
                col_names = [c["name"] for c in entry["columns"]]
                st.markdown(
                    f"**`{entry['table_name']}`** — "
                    f"{entry['row_count']} rows | "
                    f"*{entry['source']} / {entry['sheet']}*"
                )
                st.code(", ".join(col_names), language=None)

        excel_question = st.text_input(
            "💬 Ask a question about your Excel / CSV data:",
            key="excel_question",
            placeholder=(
                'e.g. "What is the total sales for Q1?" '
                'or "List all products with price > 100"'
            ),
        )

        if excel_question:
            with st.chat_message("user"):
                st.markdown(excel_question)

            with st.chat_message("assistant"):
                with st.spinner("Generating SQL and querying data…"):
                    t0 = time.time()
                    res = ask_excel(excel_question, ollama_model=OLLAMA_MODEL)
                    elapsed = round(time.time() - t0, 2)

                if res["error"]:
                    st.error(res["error"])
                else:
                    st.markdown(res["answer"])
                    st.markdown("---")
                    st.metric("Query time", f"{elapsed}s")

                    with st.expander("Generated SQL"):
                        st.code(res["sql"], language="sql")

                    if res["rows"]:
                        with st.expander(
                            f"Raw results ({len(res['rows'])} row(s))"
                        ):
                            import pandas as pd
                            df_result = pd.DataFrame(
                                res["rows"], columns=res["columns"]
                            )
                            st.dataframe(df_result, use_container_width=True)
                    else:
                        st.info("The query returned no rows.")
