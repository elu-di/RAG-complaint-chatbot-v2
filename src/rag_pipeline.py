"""
rag_pipeline.py -- Retrieval-Augmented Generation (RAG) Core Pipeline
CrediTrust Financial AI Decision-Support System:
1. Retriever   : Semantic search over ChromaDB vector store
2. Filter      : Minimum similarity threshold (MIN_SIMILARITY=0.55)
                 + Minimum source count (MIN_SOURCES=2) before calling LLM
3. Augmenter   : Prompt template with strict anti-hallucination guardrails
4. Generator   : Groq Cloud API (openai/gpt-oss-20b)
5. Evaluator   : Qualitative evaluation across representative business questions
"""

import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import sys
sys.stdout.reconfigure(encoding="utf-8")

# Explicitly locate the .env file in the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR     = os.path.join(ROOT_DIR, "vector_store", "chroma_db")
COLLECTION_NAME      = "creditrust_complaints"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL           = "openai/gpt-oss-20b"

# FIX 2A: Minimum cosine similarity — chunks below this score are noise.
MIN_SIMILARITY = 0.63

# FIX 2B: Minimum number of chunks that must pass the threshold before
# the LLM is called. If fewer than MIN_SOURCES chunks survive filtering,
# we return the fallback message immediately (prevents hallucination from
# 1 weakly-relevant chunk passing through).
MIN_SOURCES = 2

# Grounded System Prompt Template
PROMPT_TEMPLATE = """You are a senior financial intelligence analyst for CrediTrust Financial.
Your mission is to analyze customer complaints and provide executive-level, actionable insights for Product Managers and Compliance teams.

STRICT OPERATING RULES:
1. Use ONLY the factual evidence provided in the Retrieved Complaint Excerpts below.
2. If the excerpts do not contain sufficient evidence to answer the question, state:
   "Based on the provided customer complaints, there is not enough information to answer this question."
3. Do NOT invent, assume, or extrapolate facts not present in the context.
4. Always cite specific Complaint IDs and Company names when highlighting customer pain points.
5. Provide concise, bulleted, evidence-backed answers.

Retrieved Complaint Excerpts:
{context}

User Question:
{question}

Analytical Response:"""


class ComplaintRAG:
    def __init__(self, vector_dir: str = VECTOR_STORE_DIR, model_name: str = EMBEDDING_MODEL_NAME):
        """Initialize ChromaDB client, SentenceTransformer model, and Groq API client."""
        if not os.path.exists(vector_dir):
            raise FileNotFoundError(
                f"ChromaDB not found at: {vector_dir}. Run src/chunk_and_embed.py first!"
            )

        print(f"Connecting to ChromaDB at: {vector_dir}...")
        self.chroma_client = chromadb.PersistentClient(path=vector_dir)
        self.collection    = self.chroma_client.get_collection(name=COLLECTION_NAME)
        chunk_count = self.collection.count()
        print(f"  Loaded collection '{COLLECTION_NAME}' with {chunk_count:,} chunks.")

        print(f"Loading embedding model: {model_name}...")
        self.embed_model = SentenceTransformer(model_name)

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                f"GROQ_API_KEY not found in {ENV_PATH}!\n"
                "Please make sure you have saved the file (Ctrl+S) with: GROQ_API_KEY=gsk_..."
            )

        self.groq_client = Groq(api_key=api_key)
        print(f"RAG Pipeline initialized! (MIN_SIMILARITY={MIN_SIMILARITY}, MIN_SOURCES={MIN_SOURCES})\n")

    def retrieve(self, query: str, top_k: int = 6, product_filter: str = None) -> list:
        """
        Embed the query and retrieve top-k semantically relevant chunks.

        FIX 2A applied here: chunks below MIN_SIMILARITY are discarded.

        Args:
            query:          Natural language question.
            top_k:          Candidates fetched from ChromaDB (default 6).
            product_filter: Optional ChromaDB metadata filter by product_category.

        Returns:
            List of dicts {text, metadata, similarity} — only chunks >= MIN_SIMILARITY.
        """
        query_embedding = self.embed_model.encode([query], normalize_embeddings=True).tolist()

        where_clause = None
        if product_filter:
            where_clause = {"product_category": product_filter}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []
        filtered_out = 0
        for i in range(len(results["documents"][0])):
            similarity = round(1.0 - results["distances"][0][i], 4)

            if similarity < MIN_SIMILARITY:
                filtered_out += 1
                continue

            retrieved_chunks.append({
                "text":       results["documents"][0][i],
                "metadata":   results["metadatas"][0][i],
                "similarity": similarity,
            })

        if filtered_out > 0:
            print(f"  [Threshold Filter] Dropped {filtered_out} chunk(s) below {MIN_SIMILARITY} similarity.")

        return retrieved_chunks

    def generate_response(self, query: str, top_k: int = 6, product_filter: str = None) -> dict:
        """
        Full RAG pipeline: retrieve -> filter -> guard -> build prompt -> generate.

        FIX 2B applied here: require at least MIN_SOURCES chunks before calling LLM.
        """
        # 1. Retrieve with similarity filtering
        sources = self.retrieve(query, top_k=top_k, product_filter=product_filter)

        # 2. Guard: require minimum number of high-quality sources
        #    This prevents the LLM from hallucinating when only 1 weakly-relevant
        #    chunk slips through the similarity threshold.
        if len(sources) < MIN_SOURCES:
            print(f"  [Source Guard] Only {len(sources)} source(s) passed — need at least {MIN_SOURCES}. Returning fallback.")
            return {
                "answer": "Based on the provided customer complaints, there is not enough information to answer this question.",
                "sources": sources,  # Return what we found (even if insufficient) for transparency
            }

        # 3. Build structured context block for the prompt
        context_parts = []
        for idx, src in enumerate(sources, 1):
            m = src["metadata"]
            header = (
                f"[Source #{idx}] Complaint ID: {m.get('complaint_id')} | "
                f"Product: {m.get('product_category')} | "
                f"Company: {m.get('company')} | "
                f"Issue: {m.get('issue')}"
            )
            context_parts.append(f"{header}\nExcerpt: \"{src['text']}\"")

        context_str = "\n\n".join(context_parts)
        full_prompt = PROMPT_TEMPLATE.format(context=context_str, question=query)

        # 4. Generate via Groq LLM
        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional financial analytics assistant."},
                {"role": "user",   "content": full_prompt}
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content.strip()

        return {
            "answer":     answer,
            "sources":    sources,
            "raw_prompt": full_prompt,
        }


def run_qualitative_evaluation():
    """Evaluate RAG pipeline across 6 representative queries."""
    rag = ComplaintRAG()

    eval_questions = [
        # 1. Credit Cards
        ("Why are customers unhappy with Credit Card billing disputes and unauthorized fees?",
         "Credit Card"),

        # 2. Money Transfers
        # REPHRASED (FIX): Using complaint-style vocabulary instead of business jargon.
        # Original: "What are the major pain points users face with money transfer delays and account debits?"
        # Problem : "pain points" and "account debits" don't appear in complaint text → low similarity.
        # Fix     : Natural language that mirrors how real customers write complaints.
        ("Customer complaints regarding money transfers not arriving, transfer delays, or unauthorized transactions",
 "Money Transfer"),

        # 3. Savings Accounts
        ("Why do customers report frozen savings accounts or inability to access their funds?",
         "Savings Account"),

        # 4. Personal Loans (previously empty — Fixed by oversampling in Task 2)
        ("What complaints do borrowers file regarding personal loan payment processing and unexpected interest?",
         "Personal Loan"),

        # 5. Cross-Product Identity Theft / Fraud
        ("How do fraudsters open unauthorized accounts or compromise customer identities across services?",
         None),

        # 6. Out-of-scope trap: Mortgages removed in preprocessing — tests hallucination resistance
        ("What are the primary customer complaints regarding home mortgage foreclosures?",
         None),
    ]

    print("=" * 65)
    print("           RAG QUALITATIVE EVALUATION BENCHMARK")
    print("=" * 65)

    eval_results = []
    for q_idx, (question, p_filter) in enumerate(eval_questions, 1):
        print(f"\n--- Query {q_idx}/6: \"{question}\" (Filter: {p_filter}) ---")
        res = rag.generate_response(question, top_k=6, product_filter=p_filter)

        print("\n[AI Generated Answer]:")
        print(res["answer"])

        if res["sources"]:
            print("\n[Top Retrieved Evidence]:")
            for s_idx, s in enumerate(res["sources"][:2], 1):
                m = s["metadata"]
                print(f"  Source {s_idx}: ID {m['complaint_id']} | {m['product_category']} | Similarity: {s['similarity']}")
                print(f"    \"{s['text'][:140]}...\"")
        else:
            print("\n[No sources passed the similarity + source count threshold]")

        eval_results.append({
            "question":       question,
            "product_filter": p_filter,
            "answer":         res["answer"],
            "sources":        res["sources"][:2],
        })

    return eval_results


if __name__ == "__main__":
    run_qualitative_evaluation()
