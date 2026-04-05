# AI-Knowledge-Base-Completely-local-

<div align="center">

# Local AI Knowledge Base

### An on-premise intelligent document & data query platform  
### powered by hybrid retrieval, re-ranking, and text-to-SQL — **zero cloud dependency**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0071C5?style=for-the-badge&logo=meta&logoColor=white)](https://faiss.ai)
[![SQLite](https://img.shields.io/badge/SQLite-Text--to--SQL-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

```
Ask questions about your PDFs and spreadsheets in plain English.
Everything runs locally. Your data never leaves your machine.
```

</div>

---

## ✨ What This Does

This project gives you **two fully local question-answering pipelines** through a single Streamlit UI:

| | PDF RAG Pipeline | Excel Q&A Pipeline |
|---|---|---|
| **Input** | `.pdf` files (single, bulk, or ZIP folder) | `.xlsx` / `.xls` / `.csv` files |
| **How it works** | Hybrid vector + keyword search → re-ranking → LLM | Natural language → SQL → SQLite → LLM narration |
| **Index** | FAISS + BM25 | SQLite database |
| **Best for** | Prose documents, reports, manuals | Tabular data, datasets, financial sheets |

Both pipelines use **Ollama** for 100% local LLM inference — no API keys, no cloud, no data egress.

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────────┐
                        │       Streamlit UI           │
                        │  (sidebar upload + Q&A tabs) │
                        └────────────┬────────────────┘
                                     │
               ┌─────────────────────┴──────────────────────┐
               │                                            │
    ┌──────────▼──────────┐                    ┌────────────▼───────────┐
    │   PDF RAG Pipeline   │                    │   Excel Q&A Pipeline   │
    │                      │                    │                        │
    │  pdfplumber          │                    │  pandas                │
    │  → text + tables     │                    │  → load sheets         │
    │                      │                    │                        │
    │  SentenceTransformer │                    │  SQLite                │
    │  → 384-dim vectors   │                    │  → one table/sheet     │
    │                      │                    │                        │
    │  FAISS (dense)       │                    │  Schema catalogue      │
    │  + BM25 (sparse)     │                    │  → column types        │
    │  → hybrid search     │                    │  + sample values       │
    │                      │                    │                        │
    │  CrossEncoder        │                    │  Ollama                │
    │  → re-rank top-K     │                    │  → SQL generation      │
    └──────────┬──────────┘                    └────────────┬───────────┘
               │                                            │
               └─────────────────────┬──────────────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │   Ollama  (llama3.2:1b)  │
                        │   Local LLM inference    │
                        │   Zero data egress        │
                        └─────────────────────────┘
```

---

## 🚀 Features

- **🔀 Hybrid Search** — Fuses dense vector (FAISS cosine) and sparse keyword (BM25) retrieval with a 60/40 weighted blend for best-of-both recall
- **📊 Table-Aware PDF Ingestion** — `pdfplumber` extracts table rows as structured chunks; numeric queries get an automatic score boost
- **🎯 Cross-Encoder Re-ranking** — `ms-marco-MiniLM-L-6-v2` re-scores every candidate `(query, chunk)` pair — far more precise than embedding similarity alone
- **🗃️ Text-to-SQL for Excel** — Natural language is converted to a SQLite `SELECT` via a schema-grounded LLM prompt, executed, then narrated back in plain English
- **📁 Folder Upload Support** — Upload a ZIP archive to ingest an entire folder of PDFs or spreadsheets in one shot, preserving sub-folder structure
- **✅ Answer Verification** — A second LLM pass checks whether the answer is grounded in the retrieved context and flags unsupported claims
- **🔒 Fully Private** — Ollama runs entirely on-premise; compatible with air-gapped environments

---

## 📂 Project Structure

```
.
├── app.py                  # Streamlit UI — sidebar upload + dual Q&A tabs
├── ingest_local.py         # PDF ingestion → FAISS + BM25 index builder
├── chat_local.py           # PDF Q&A engine — hybrid search + rerank + LLM
├── ingest_excel.py         # Excel/CSV ingestion → SQLite + schema catalogue
├── excel_qa.py             # Text-to-SQL engine — SQL gen + execution + narration
├── db.py                   # Optional MongoDB connector for chunk persistence
├── data/                   # Drop your PDFs and spreadsheets here
│   └── ...
├── faiss_index/
│   ├── index.faiss         # FAISS vector index (auto-generated)
│   └── metadata.pkl        # BM25 model + chunk metadata (auto-generated)
└── excel_index/
    ├── excel.db            # SQLite database (auto-generated)
    └── schema.json         # Table schema catalogue (auto-generated)
```

---

## ⚙️ Installation

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Pull a model:
  ```bash
  ollama pull llama3.2:1b
  ```

### 2. Clone & install dependencies

```bash
git clone https://github.com/your-username/local-rag-excel-qa.git
cd local-rag-excel-qa

pip install -r requirements.txt
```

<details>
<summary><b>requirements.txt (click to expand)</b></summary>

```
streamlit
python-dotenv
faiss-cpu
sentence-transformers
rank-bm25
pdfplumber
nltk
pandas
openpyxl
langchain-ollama
langchain-core
pymongo          # optional — only needed for MongoDB persistence
```

</details>

### 3. Configure environment

```bash
cp .env.example .env
```

```env
# .env
OLLAMA_MODEL=llama3.2:1b
TOP_K=3
# MONGODB_URI=mongodb://localhost:27017   # optional
```

### 4. Launch

```bash
streamlit run app.py
```

---

## 📖 Usage

### PDF Q&A

1. Open the **sidebar → 📄 PDFs** tab
2. Upload one or more `.pdf` files, **or** upload a `.zip` archive of a folder
3. Click **⚙️ Build PDF Index** — embeddings are computed once and cached
4. Switch to the **"Ask your PDFs"** main tab
5. Type your question in plain English

### Excel / CSV Q&A

1. Open the **sidebar → 📊 Excel / CSV** tab
2. Upload `.xlsx`, `.xls`, or `.csv` files (or a ZIP of them)
3. Click **⚙️ Build Excel Index** — sheets are loaded into SQLite
4. Switch to the **"Ask your Excel data"** main tab
5. Ask anything — aggregations, filters, lookups, comparisons

> **Tip:** The schema reference expander shows every indexed table with column names and types — useful for phrasing your question precisely.

---

## 🧠 How the PDF Pipeline Works

```
Query
  │
  ├─► BM25 (sparse)   → top-12 keyword matches   ─┐
  │                                                 ├─► Candidate pool (union)
  └─► FAISS (dense)   → top-12 vector matches    ─┘
                                                      │
                                              Hybrid fusion score
                                          (α=0.6 dense + β=0.4 sparse)
                                              + table chunk boost (+0.15)
                                                      │
                                              CrossEncoder rerank
                                                      │
                                              Top-3 chunks
                                                      │
                                              Ollama → Answer
                                                      │
                                              Verification pass
```

---

## 🗄️ How the Excel Pipeline Works

```
Question: "What is the total revenue for the North region in Q1?"
    │
    ▼
Schema catalogue (table names, columns, types, sample values)
    │
    ▼
Ollama → generates SQL:
    SELECT SUM(revenue)
    FROM sales_Sheet1
    WHERE region = 'North' AND quarter = 'Q1'
    │
    ▼
SQLite executes query → result rows
    │
    ▼
Ollama → narrates: "The total Q1 revenue for the North region is $482,300..."
```

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:1b` | Ollama model to use for generation |
| `TOP_K` | `3` | Number of chunks sent to LLM after reranking |
| `MONGODB_URI` | *(unset)* | MongoDB connection string (optional persistence) |

**Retrieval tuning** (in `chat_local.py`):

| Parameter | Value | Description |
|---|---|---|
| `k_dense` | `12` | FAISS candidates per query |
| `k_sparse` | `12` | BM25 candidates per query |
| `alpha` | `0.6` | Dense retrieval weight |
| `beta` | `0.4` | Sparse retrieval weight |
| `window` | `1` | Chunk expansion window (±1 neighbor) |
| Table boost | `+0.15` | Score bonus for table-type chunks |

---

## 🛣️ Roadmap

- [ ] Multi-turn conversation memory
- [ ] Source citation with clickable page number links
- [ ] Automatic pipeline routing (auto-detect PDF vs Excel question)
- [ ] Support for `.docx` and `.pptx` documents
- [ ] Role-based document access control
- [ ] GPU-accelerated embedding inference
- [ ] REST API for integration with external tools
- [ ] Larger model support (`llama3:8b`, `mistral`, `mixtral`)

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change. For pull requests:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built entirely in-house · Zero cloud dependency · Production-ready**

*If this project helped you, consider giving it a ⭐*

</div>
