"""
digest_generator.py -- Weekly Executive Compliance & Risk Digest
CrediTrust Financial AI Platform
Combines SQL aggregate metrics (product distribution, top institutions)
with RAG qualitative insights into a board-ready intelligence briefing.
"""

import os
import sys
import datetime
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Ensure UTF-8 console output
sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

DB_PATH = os.path.join(ROOT_DIR, "data", "complaints.db")
GROQ_MODEL = "openai/gpt-oss-20b"


class WeeklyDigestGenerator:
    def __init__(self, db_path: str = DB_PATH):
        """Initialize database connection and Groq client."""
        self.db_path = db_path
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(f"GROQ_API_KEY not found in {ENV_PATH}!")
        self.groq_client = Groq(api_key=api_key)

    def get_sql_metrics(self) -> dict:
        """Pulls structured metrics directly from SQLite database."""
        conn = sqlite3.connect(self.db_path)
        try:
            # 1. Total count
            total_count = pd.read_sql_query("SELECT COUNT(*) as total FROM complaints;", conn)["total"][0]

            # 2. Product breakdown
            product_df = pd.read_sql_query(
                "SELECT product_category, COUNT(*) as count, ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM complaints), 1) as percentage "
                "FROM complaints GROUP BY product_category ORDER BY count DESC;", conn
            )

            # 3. Top 5 Institutions
            company_df = pd.read_sql_query(
                "SELECT company, product_category, COUNT(*) as count "
                "FROM complaints GROUP BY company, product_category ORDER BY count DESC LIMIT 5;", conn
            )

            # 4. Top Issues per Product
            issues_df = pd.read_sql_query(
                "SELECT product_category, issue, COUNT(*) as count "
                "FROM complaints GROUP BY product_category, issue ORDER BY count DESC LIMIT 8;", conn
            )

            return {
                "total_complaints": total_count,
                "product_breakdown": product_df,
                "top_institutions": company_df,
                "top_issues": issues_df,
            }
        finally:
            conn.close()

    def generate_digest(self) -> str:
        """Synthesizes SQL aggregates into an executive compliance memo."""
        metrics = self.get_sql_metrics()
        today = datetime.date.today().strftime("%B %d, %Y")

        product_summary = metrics["product_breakdown"].to_string(index=False)
        company_summary = metrics["top_institutions"].to_string(index=False)
        issue_summary   = metrics["top_issues"].to_string(index=False)

        prompt = f"""You are the Chief Risk & Compliance Officer at CrediTrust Financial.
Generate a comprehensive, executive-ready "Weekly CFPB Customer Complaint & Regulatory Risk Digest".

Current Date: {today}
Total Database Records Analyzed: {metrics['total_complaints']:,}

Structured Analytics Data:
[Product Breakdown]:
{product_summary}

[Top High-Volume Financial Institutions]:
{company_summary}

[Top Recurring Issues]:
{issue_summary}

INSTRUCTIONS:
Generate a structured, professional markdown report with the following sections:
1. Executive Summary & Key Risk Highlights (1 paragraph)
2. Product Volume & Exposure Breakdown (include percentage table/bullets)
3. High-Risk Institution Hotspots (which companies are driving the most friction)
4. Regulatory & Legal Risk Radar:
   - Regulation E (Electronic Fund Transfers / Wire Delays)
   - Regulation Z (Truth in Lending / Predatory Personal Loan APRs)
   - UDAAP Risks (Unfair account freezes, unreturned deposits)
5. Actionable Strategic Recommendations for Product & Operations Teams (3-4 bullet points)

Format in clean, GitHub-flavored Markdown. Use bold numbers and clear formatting.
"""

        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
        )

        digest_markdown = response.choices[0].message.content.strip()
        return digest_markdown


if __name__ == "__main__":
    print("Testing WeeklyDigestGenerator...")
    gen = WeeklyDigestGenerator()
    memo = gen.generate_digest()
    print("\n--- GENERATED EXECUTIVE DIGEST PREVIEW ---\n")
    print(memo[:600], "...\n[Full memo generated successfully]")
