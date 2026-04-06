"""
ingest_excel.py
───────────────
Loads .xlsx / .xls / .csv files and:
  1. Persists every sheet as a table inside a single SQLite database
     (excel_index/excel.db) so SQL queries can run against them later.
  2. Saves a JSON schema catalogue (excel_index/schema.json) that the
     text-to-SQL engine uses to build accurate prompts.

Run standalone:
    python ingest_excel.py
"""

import os
import json
import sqlite3
import re

import pandas as pd

EXCEL_INDEX_DIR = "excel_index"
DB_PATH = os.path.join(EXCEL_INDEX_DIR, "excel.db")
SCHEMA_PATH = os.path.join(EXCEL_INDEX_DIR, "schema.json")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    """Convert an arbitrary string into a safe SQLite identifier."""
    name = re.sub(r"[^\w]", "_", str(name).strip())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name or name[0].isdigit():
        name = "col_" + name
    return name[:60]


def _unique_table_name(existing: set, base: str) -> str:
    base = _sanitize(base)[:50]
    candidate = base
    i = 2
    while candidate in existing:
        candidate = f"{base}_{i}"
        i += 1
    return candidate


def _infer_col_types(df: pd.DataFrame) -> dict:
    """Return a dict of {col_name: sqlite_type_string}."""
    mapping = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            mapping[col] = "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            mapping[col] = "REAL"
        elif pd.api.types.is_bool_dtype(dtype):
            mapping[col] = "INTEGER"
        else:
            mapping[col] = "TEXT"
    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# Core ingest
# ──────────────────────────────────────────────────────────────────────────────

def load_excel_files(folder: str = "data") -> list[dict]:
    """
    Walk *folder* recursively.  Load every .xlsx / .xls / .csv file.
    Returns a list of sheet-level dicts:
        {source, sheet, table_name, df}
    """
    records = []
    existing_table_names: set[str] = set()

    for root, _, files in os.walk(folder):
        for fname in files:
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, folder)
            ext = fname.lower().split(".")[-1]

            try:
                if ext in ("xlsx", "xls"):
                    sheets: dict[str, pd.DataFrame] = pd.read_excel(
                        path, sheet_name=None, dtype=str
                    )
                elif ext == "csv":
                    sheets = {fname: pd.read_csv(path, dtype=str)}
                else:
                    continue

                for sheet_name, df in sheets.items():
                    if df.empty:
                        continue

                    # Clean column names
                    df.columns = [_sanitize(c) for c in df.columns]

                    # Drop fully-empty rows / columns
                    df = df.dropna(how="all").dropna(axis=1, how="all")

                    # Convert numeric-looking columns back to proper types
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except (ValueError, TypeError):
                            pass

                    base = f"{os.path.splitext(fname)[0]}_{_sanitize(sheet_name)}"
                    tname = _unique_table_name(existing_table_names, base)
                    existing_table_names.add(tname)

                    records.append(
                        {
                            "source": rel,
                            "sheet": sheet_name,
                            "table_name": tname,
                            "df": df,
                        }
                    )

            except Exception as e:
                print(f"⚠ Could not load {path}: {e}")

    return records


def build_excel_index(folder: str = "data") -> int:
    """
    (Re-)build the SQLite database and schema catalogue.
    Returns the number of tables written.
    """
    os.makedirs(EXCEL_INDEX_DIR, exist_ok=True)

    records = load_excel_files(folder)
    if not records:
        print(f"⚠ No Excel/CSV files found under '{folder}'.")
        return 0

    # ── Write SQLite ───────────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    # Drop ALL existing tables first so a rebuild is clean
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for (t,) in cur.fetchall():
        cur.execute(f'DROP TABLE IF EXISTS "{t}"')
    conn.commit()

    schema_catalogue = []

    for rec in records:
        df: pd.DataFrame = rec["df"]
        tname = rec["table_name"]

        df.to_sql(tname, conn, if_exists="replace", index=False)

        # Sample rows for the schema catalogue (helps the LLM)
        sample_rows = df.head(3).fillna("").to_dict(orient="records")

        col_types = _infer_col_types(df)

        schema_catalogue.append(
            {
                "source": rec["source"],
                "sheet": rec["sheet"],
                "table_name": tname,
                "row_count": len(df),
                "columns": [
                    {
                        "name": col,
                        "type": col_types.get(col, "TEXT"),
                        "sample_values": df[col].dropna().astype(str).unique()[:5].tolist(),
                    }
                    for col in df.columns
                ],
                "sample_rows": sample_rows,
            }
        )

    conn.close()

    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema_catalogue, f, indent=2, default=str)

    print(f"✔ Excel index built: {len(records)} table(s) → {DB_PATH}")
    return len(records)


# ──────────────────────────────────────────────────────────────────────────────
# Public accessors used by excel_qa.py
# ──────────────────────────────────────────────────────────────────────────────

def get_schema() -> list[dict]:
    """Load the schema catalogue from disk."""
    if not os.path.exists(SCHEMA_PATH):
        return []
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def get_db_connection() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Excel DB not found at {DB_PATH}. Run build_excel_index() first.")
    return sqlite3.connect(DB_PATH)


def excel_index_exists() -> bool:
    return os.path.exists(DB_PATH) and os.path.exists(SCHEMA_PATH)


if __name__ == "__main__":
    build_excel_index()
