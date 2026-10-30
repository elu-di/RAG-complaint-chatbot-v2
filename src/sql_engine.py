"""
sql_engine.py -- Structured Analytics & Text-to-SQL Engine
CrediTrust Financial AI Decision-Support Platform
Converts natural language quantitative questions into optimized SQLite queries,
executes them safely on the complaint database, and synthesizes executive insights.
"""

import os
import sys
import re
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Configure UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8")

# Explicitly locate environment
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

DB_PATH  = os.path.join(ROOT_DIR, "data", "complaints.db")
CSV_PATH = os.path.join(ROOT_DIR, "data", "stratified_sample.csv")
GROQ_MODEL = "openai/gpt-oss-20b"

# Safe Schema Context for Groq LLM
SQL_PROMPT_TEMPLATE = """You are an expert SQL analytics engineer for CrediTrust Financial.
Convert the user's natural language business question into an optimized, valid SQLite SELECT query.

Database Schema:
Table Name: complaints
Columns:
- complaint_id (INTEGER) : Unique identifier
- date_received (TEXT)    : Date format 'YYYY-MM-DD'
- product_category (TEXT) : 'Credit Card', 'Personal Loan', 'Savings Account', or 'Money Transfer'
- issue (TEXT)            : High-level complaint issue (e.g., 'Improper fees', 'Trouble during payment process')
- sub_issue (TEXT)        : Detailed specific complaint issue
- company (TEXT)          : Financial institution / company name (e.g., 'JPMorgan Chase & Co.', 'Citibank, N.A.')
- state (TEXT)            : 2-letter US state code ('CA', 'NY', 'TX', 'FL', etc.)

STRICT RULES:
1. Output ONLY the raw SQL query. Never include markdown code fences like ```sql or explanations.
2. Only write read-only SELECT queries. Never use INSERT, UPDATE, DELETE, DROP, or ALTER.
3. For rankings or totals, use COUNT(*) with GROUP BY and ORDER BY count DESC.
4. Always include a sensible LIMIT (default 10) unless computing an aggregate scalar (like total count).
5. Use LIKE or lower() for partial string matches where appropriate.

User Question: {question}
SQL:"""

SYNTHESIS_PROMPT_TEMPLATE = """You are a senior financial analyst at CrediTrust Financial.
Synthesize the following SQL query results into an executive-level, analytical insight.

User Question: {question}
SQL Query Executed: {sql_query}

Query Results:
{data_table}

INSTRUCTIONS:
1. Provide a concise, bulleted executive takeaway highlighting the key patterns, percentages, or top institutions.
2. Include specific numbers and percentages where relevant.
3. Keep the tone professional, objective, and actionable for Compliance and Product Managers.
"""


def initialize_complaints_db(db_path: str = DB_PATH, csv_path: str = CSV_PATH):
    """Creates SQLite database and builds performance indexes if not already present."""
    if os.path.exists(db_path):
        return db_path

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source dataset not found at {csv_path} to build SQLite database.")

    print(f"Building SQLite database at: {db_path}...")
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    df.to_sql("complaints", conn, index=False, if_exists="replace")

    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_product ON complaints(product_category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_company ON complaints(company);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_issue ON complaints(issue);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON complaints(date_received);")
    conn.commit()
    conn.close()
    print(f"SQLite database initialized with {len(df):,} records and indexed.")
    return db_path


class ComplaintSQL:
    def __init__(self, db_path: str = DB_PATH):
        """Initialize database connection and Groq client."""
        self.db_path = initialize_complaints_db(db_path)
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(f"GROQ_API_KEY not found in {ENV_PATH}!")

        self.groq_client = Groq(api_key=api_key)

    def generate_sql(self, question: str) -> str:
        """Translates natural language questions to safe SQLite SELECT statements."""
        prompt = SQL_PROMPT_TEMPLATE.format(question=question)
        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=250,
        )

        sql = response.choices[0].message.content.strip()
        sql = re.sub(r"```(sql)?", "", sql, flags=re.IGNORECASE).strip()
        return sql

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Executes query with strict security validation.
        Blocks harmful statements (DROP, DELETE, UPDATE, INSERT, ALTER).
        """
        cleaned_sql = sql_query.strip()
        normalized = cleaned_sql.upper()

        # Security check: Read-only enforcer
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "ATTACH", "TRUNCATE"]
        for kw in forbidden_keywords:
            if re.search(r"\b" + kw + r"\b", normalized):
                raise ValueError(f"Security Alert: Execution blocked. '{kw}' is not permitted.")

        if not normalized.startswith("SELECT") and not normalized.startswith("WITH"):
            raise ValueError("Security Alert: Only SELECT or WITH queries are permitted.")

        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(cleaned_sql, conn)
            return df
        finally:
            conn.close()

    def synthesize_insight(self, question: str, sql_query: str, df: pd.DataFrame) -> str:
        """Converts query data table into an executive summary via Groq LLM."""
        if df.empty:
            return "No matching records found in the database for this query."

        data_str = df.to_string(index=False)
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            question=question,
            sql_query=sql_query,
            data_table=data_str
        )

        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
        )

        return response.choices[0].message.content.strip()

    def query_and_analyze(self, question: str) -> dict:
        """
        Complete end-to-end pipeline:
        Natural language -> SQL -> SQLite execution -> Executive synthesis.
        """
        sql_query = self.generate_sql(question)
        df = self.execute_query(sql_query)
        synthesis = self.synthesize_insight(question, sql_query, df)

        return {
            "question":   question,
            "sql_query":  sql_query,
            "data":       df,
            "synthesis":  synthesis,
        }


if __name__ == "__main__":
    print("Testing ComplaintSQL Engine...")
    engine = ComplaintSQL()
    test_q = "Which 5 companies have the highest number of complaints?"
    result = engine.query_and_analyze(test_q)
    print("\n[SQL Query]:", result["sql_query"])
    print("\n[Data Table]:\n", result["data"])
    print("\n[Executive Synthesis]:\n", result["synthesis"])
