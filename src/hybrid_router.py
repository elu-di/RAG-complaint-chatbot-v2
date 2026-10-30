"""
hybrid_router.py -- Intelligent Intent Router for CrediTrust Financial
Dispatches user questions dynamically between:
1. SQL Analytics Engine (for counts, rankings, percentages, and volume metrics)
2. Semantic Vector RAG Engine (for root causes, customer sentiment, and narrative explanations)
"""

import os
import sys
import re
from dotenv import load_dotenv
from groq import Groq

# Ensure UTF-8 console output
sys.stdout.reconfigure(encoding="utf-8")

# Explicitly locate environment
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

from rag_pipeline import ComplaintRAG
from sql_engine import ComplaintSQL

GROQ_MODEL = "openai/gpt-oss-20b"

ROUTER_PROMPT_TEMPLATE = """You are an intent routing classifier for CrediTrust Financial AI.
Analyze the user's question and determine the single best execution engine:
1. SQL: Use when the user asks for numbers, counts, rankings, top institutions, volume, breakdown, "which companies have the most", or quantitative statistics.
2. RAG: Use when the user asks "why", "how", explaining customer experiences, pain points, fraud methods, or narrative reasons for complaints.

Output ONLY the word 'SQL' or 'RAG'. No extra text.

Question: {question}
Intent:"""


class HybridRouter:
    def __init__(self, rag_engine: ComplaintRAG = None, sql_engine: ComplaintSQL = None):
        """Initializes both sub-engines and Groq classifier."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(f"GROQ_API_KEY not found in {ENV_PATH}!")

        self.groq_client = Groq(api_key=api_key)
        self.rag = rag_engine if rag_engine is not None else ComplaintRAG()
        self.sql = sql_engine if sql_engine is not None else ComplaintSQL()

    def classify_intent(self, question: str) -> str:
        """
        Fast hybrid classification:
        1. Fast regex pattern check (0ms latency)
        2. LLM fallback for nuanced/ambiguous questions
        """
        q_lower = question.lower()

        # Quantitative indicators -> SQL
        sql_patterns = [
            r"\bhow many\b", r"\btop\s*\d+\b", r"\bmost common\b", r"\bmost complained\b",
            r"\bhighest\b", r"\branking\b", r"\bpercentage\b", r"\bproportion\b",
            r"\bcount\b", r"\bbreakdown\b", r"\bvolume\b", r"\btotal number\b",
            r"\bwhich\s*(\d+\s*)?compan(y|ies)\b", r"\bwhich\s*product\s*has\s*(the\s*)?most\b",
            r"\bhow often\b", r"\brank\b"
        ]
        for pat in sql_patterns:
            if re.search(pat, q_lower):
                return "SQL"

        # Qualitative indicators -> RAG
        rag_patterns = [
            r"\bwhy\b", r"\bhow do\b", r"\bexplain\b", r"\breason\b",
            r"\bpain points\b", r"\bwhat happened\b", r"\bdescribe\b",
            r"\bexperience\b", r"\bstruggle\b", r"\bfraudulent scheme\b"
        ]
        for pat in rag_patterns:
            if re.search(pat, q_lower):
                return "RAG"

        # Fallback to LLM classification
        prompt = ROUTER_PROMPT_TEMPLATE.format(question=question)
        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )

        intent = response.choices[0].message.content.strip().upper()
        return "SQL" if "SQL" in intent else "RAG"

    def dispatch(self, question: str, product_filter: str = None, min_similarity: float = 0.50, top_k: int = 6) -> dict:
        """
        Routes and executes the question using the most appropriate engine.
        Returns a unified dictionary with structured data and answers.
        """
        intent = self.classify_intent(question)

        if intent == "SQL":
            sql_res = self.sql.query_and_analyze(question)
            return {
                "intent":      "SQL",
                "question":    question,
                "sql_query":   sql_res["sql_query"],
                "data":        sql_res["data"],
                "answer":      sql_res["synthesis"],
                "sources":     [],
                "guard_active": False,
            }
        else:
            rag_res = self.rag.generate_response(
                query=question,
                top_k=top_k,
                product_filter=product_filter,
                min_similarity=min_similarity
            )
            return {
                "intent":      "RAG",
                "question":    question,
                "sql_query":   None,
                "data":        None,
                "answer":      rag_res["answer"],
                "sources":     rag_res["sources"],
                "guard_active": len(rag_res["sources"]) < 2,
            }


if __name__ == "__main__":
    print("Testing HybridRouter with enhanced patterns...")
    router = HybridRouter()

    test_queries = [
        "Which 3 companies have the most Money Transfer complaints?",
        "Why do customers complain about unexpected personal loan interest?",
        "How many complaints were filed for credit card billing disputes?",
    ]

    for q in test_queries:
        print(f"\nEvaluating: '{q}'")
        classified = router.classify_intent(q)
        print(f"  Classified Intent: {classified}")
        res = router.dispatch(q)
        print(f"  Engine Used: {res['intent']}")
        if res['intent'] == 'SQL':
            print("  SQL Query:", res['sql_query'])
            print(f"  Data Table Rows: {len(res['data'])}")
        else:
            print(f"  Sources Retrieved: {len(res['sources'])}")
        print("  Answer Preview:", res['answer'][:120], "...")
