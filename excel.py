"""
excel_qa.py
───────────
Text-to-SQL question-answering over Excel/CSV files.

Pipeline
────────
1. Load schema catalogue → build a compact schema string for the prompt.
2. Ask Ollama to generate a SQLite SQL query given the question + schema.
3. Execute the query against the SQLite DB.
4. Ask Ollama to narrate the result rows as a natural-language answer.

Public API
──────────
    answer = ask_excel(question, ollama_model="llama3.2:1b")
    # Returns dict: {sql, rows, columns, answer, error}
"""

import os
import re
import sqlite3
import json
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ingest_excel import get_schema, get_db_connection, excel_index_exists

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

# ──────────────────────────────────────────────────────────────────────────────
# Schema formatting
# ──────────────────────────────────────────────────────────────────────────────

def _format_schema(schema: list[dict], max_tables: int = 10) -> str:
    """
    Produce a compact schema description.  Keeps the prompt short even when
    many sheets are loaded.
    """
    lines = []
    for entry in schema[:max_tables]:
        tname = entry["table_name"]
        source_hint = f"{entry['source']} / {entry['sheet']}"
        cols_desc = ", ".join(
            f"{c['name']} ({c['type']})" for c in entry["columns"]
        )
        lines.append(f"TABLE `{tname}`  -- from: {source_hint}")
        lines.append(f"  Columns: {cols_desc}")
        lines.append(f"  Rows: {entry['row_count']}")

        # Add a sample row so the LLM understands value format
        if entry.get("sample_rows"):
            sample = entry["sample_rows"][0]
            sample_str = ", ".join(f"{k}={repr(v)}" for k, v in list(sample.items())[:6])
            lines.append(f"  Sample: {sample_str}")
        lines.append("")

    if len(schema) > max_tables:
        lines.append(f"... and {len(schema) - max_tables} more table(s).")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# SQL extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_sql(text: str) -> str:
    """
    Pull the first SQL statement out of an LLM response.
    Handles markdown code blocks, raw SQL, etc.
    """
    # Try ```sql ... ``` first
    m = re.search(r"```(?:sql)?\s*(SELECT[\s\S]+?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Try bare SELECT ... ; or SELECT ... to end-of-string
    m = re.search(r"(SELECT[\s\S]+?)(;|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

SQL_SYSTEM = """You are an expert SQLite query writer.
Given a database schema and a natural-language question, write ONE valid SQLite SELECT statement that answers the question.

Rules:
- Use ONLY table and column names that appear in the schema.
- Wrap column/table names in double-quotes if they contain spaces or special characters.
- Use LIKE for partial text matches (case-insensitive with LOWER()).
- Do NOT use any DDL (CREATE, DROP, INSERT, UPDATE, DELETE).
- Return ONLY the SQL query — no explanation, no markdown, no preamble.
"""

SQL_HUMAN = """Schema:
{schema}

Question: {question}

SQL:"""

NARRATE_SYSTEM = """You are a helpful data analyst.
Given a SQL query, its result, and the original question, produce a clear, concise natural-language answer.
If the result is empty, say so clearly and suggest why.
Do NOT repeat the SQL.  Be direct.
"""

NARRATE_HUMAN = """Question: {question}

SQL used: {sql}

Result ({row_count} row(s)):
{result_table}

Answer:"""


# ──────────────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────────────

def ask_excel(
    question: str,
    ollama_model: str = OLLAMA_MODEL,
    max_rows: int = 200,
) -> dict:
    """
    Answer a question about the loaded Excel data using text-to-SQL.

    Returns:
        {
            "sql":     str,          # generated SQL
            "columns": list[str],    # result column names
            "rows":    list[tuple],  # result rows
            "answer":  str,          # natural-language answer
            "error":   str | None,   # error message if something failed
        }
    """
    result = {"sql": "", "columns": [], "rows": [], "answer": "", "error": None}

    if not excel_index_exists():
        result["error"] = "No Excel index found. Please upload and index Excel files first."
        return result

    schema = get_schema()
    if not schema:
        result["error"] = "Schema catalogue is empty."
        return result

    schema_str = _format_schema(schema)
    llm = ChatOllama(model=ollama_model, temperature=0)

    # ── Step 1: generate SQL ──────────────────────────────────────────────
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", SQL_SYSTEM),
        ("human", SQL_HUMAN),
    ])
    sql_chain = sql_prompt | llm | StrOutputParser()

    raw_sql_response = sql_chain.invoke({"schema": schema_str, "question": question})
    sql = _extract_sql(raw_sql_response)
    result["sql"] = sql

    # ── Step 2: execute SQL ───────────────────────────────────────────────
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(max_rows)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        conn.close()

        result["columns"] = columns
        result["rows"] = rows

    except sqlite3.Error as e:
        result["error"] = f"SQL execution error: {e}\n\nGenerated SQL:\n{sql}"
        return result

    # ── Step 3: narrate result ────────────────────────────────────────────
    if not rows:
        result_table = "(no rows returned)"
    else:
        # Simple ASCII table for the prompt
        header = " | ".join(columns)
        separator = "-" * len(header)
        data_rows = "\n".join(" | ".join(str(v) for v in row) for row in rows[:20])
        result_table = f"{header}\n{separator}\n{data_rows}"
        if len(rows) > 20:
            result_table += f"\n... ({len(rows)} rows total, showing first 20)"

    narrate_prompt = ChatPromptTemplate.from_messages([
        ("system", NARRATE_SYSTEM),
        ("human", NARRATE_HUMAN),
    ])
    narrate_chain = narrate_prompt | llm | StrOutputParser()

    answer = narrate_chain.invoke({
        "question": question,
        "sql": sql,
        "row_count": len(rows),
        "result_table": result_table,
    })
    result["answer"] = answer

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not excel_index_exists():
        print("Run ingest_excel.py first to build the index.")
    else:
        print("\n Excel Q&A ready (type 'exit' to quit)\n")
        while True:
            q = input("Q: ").strip()
            if q.lower() == "exit":
                break
            res = ask_excel(q)
            if res["error"]:
                print(f"Error: {res['error']}")
            else:
                print(f"\nSQL: {res['sql']}")
                print(f"\nAnswer: {res['answer']}\n")
