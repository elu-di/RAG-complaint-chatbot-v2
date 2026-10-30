"""
chunk_and_embed.py -- Text Chunking, Embedding, and ChromaDB Indexing
Pipeline for CrediTrust Financial RAG Chatbot:
1. Performs Oversampled Stratified Sampling (min 2,500 per product, max 4,000)
2. Chunks narratives using RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
3. Generates 384-dimensional embeddings via sentence-transformers/all-MiniLM-L6-v2
4. Persists chunks + embeddings + rich metadata in ChromaDB (vector_store/chroma_db)
5. Executes a semantic test query to verify retrieval quality
"""

import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH        = os.path.join(ROOT_DIR, "data", "filtered_complaints.csv")
SAMPLE_PATH      = os.path.join(ROOT_DIR, "data", "stratified_sample.csv")
VECTOR_STORE_DIR = os.path.join(ROOT_DIR, "vector_store", "chroma_db")
COLLECTION_NAME  = "creditrust_complaints"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# FIX 1: Oversampled quotas - every product guaranteed up to MAX_PER_PRODUCT,
# so minority classes (Personal Loan, Money Transfer) are no longer starved.
MAX_PER_PRODUCT = 4_000


def get_oversampled_stratified_sample(
    input_path: str = DATA_PATH,
    output_path: str = SAMPLE_PATH,
    max_per_product: int = MAX_PER_PRODUCT,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Oversampled Stratified Sampling Strategy
    ----------------------------------------
    Problem with pure proportional sampling:
      - Credit Card has 59% of records -> dominates the entire index.
      - Personal Loan has only 4.3%   -> barely represented, poor retrieval.

    Solution: per-product quota cap.
      - Each product gets min(available_records, max_per_product).
      - This deliberately equalizes minority classes to improve retrieval.
      - Same logic as class-balanced sampling in ML classification tasks.
    """
    print(f"Loading filtered complaints from: {input_path}")
    df = pd.read_csv(input_path)
    total_records = len(df)
    print(f"Total available complaints: {total_records:,}")
    print(f"\nSampling Strategy: per-product cap (max={max_per_product:,} each)\n")

    product_samples = []
    for product, group in df.groupby("product_category"):
        available  = len(group)
        n_to_sample = min(available, max_per_product)
        sampled = group.sample(n=n_to_sample, random_state=random_state)
        product_samples.append(sampled)
        print(f"  {product:<30}: {available:>7,} available  ->  sampled {n_to_sample:>5,}")

    # Concatenate and shuffle so products are interleaved (no ordering bias)
    sample_df = (
        pd.concat(product_samples, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    print(f"\nTotal oversampled records : {len(sample_df):,}")
    print("\nFinal breakdown by Product Category:")
    for p, c in sample_df["product_category"].value_counts().items():
        pct = (c / len(sample_df)) * 100
        print(f"  - {p:<30}: {c:>6,} ({pct:.1f}%)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved oversampled sample to: {output_path}")
    return sample_df


def chunk_complaints(
    df: pd.DataFrame,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list:
    """Split complaint narratives into overlapping chunks with traceable metadata."""
    print(f"\nChunking narratives (chunk_size={chunk_size}, overlap={chunk_overlap})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        text = str(row.get("cleaned_narrative", ""))
        if not text.strip():
            continue

        raw_chunks = splitter.split_text(text)
        total_chunks = len(raw_chunks)

        for idx, piece in enumerate(raw_chunks):
            chunk_id = f"{row['complaint_id']}_{idx}"
            metadata = {
                "complaint_id":     str(row.get("complaint_id", "")),
                "product_category": str(row.get("product_category", "")),
                "issue":            str(row.get("issue", ""))[:150],
                "company":          str(row.get("company", ""))[:100],
                "state":            str(row.get("state", "")),
                "date_received":    str(row.get("date_received", "")),
                "chunk_index":      int(idx),
                "total_chunks":     int(total_chunks),
            }
            chunks.append({"id": chunk_id, "text": piece, "metadata": metadata})

    avg = len(chunks) / len(df)
    print(f"Total chunks produced: {len(chunks):,} (Average: {avg:.1f} per complaint)")
    return chunks


def build_and_persist_vector_store(
    chunks: list,
    persist_dir: str = VECTOR_STORE_DIR,
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = 500,
):
    """Generate embeddings and index into ChromaDB."""
    print(f"\nLoading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    os.makedirs(persist_dir, exist_ok=True)
    print(f"Initializing persistent ChromaDB client at: {persist_dir}")
    client = chromadb.PersistentClient(path=persist_dir)

    # Drop old collection for a clean rebuild
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Dropped existing ChromaDB collection for clean rebuild.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Embedding and indexing {len(chunks):,} chunks in batches of {batch_size}...")
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(chunks), batch_size), total=total_batches, desc="Indexing"):
        batch     = chunks[i : i + batch_size]
        texts     = [c["text"]     for c in batch]
        ids       = [c["id"]       for c in batch]
        metadatas = [c["metadata"] for c in batch]

        embeddings = model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(f"\nSuccessfully indexed {collection.count():,} chunks in ChromaDB!")
    return client, collection, model


def test_semantic_search(
    collection, model,
    query: str = "unexpected interest charges on my personal loan",
    top_k: int = 3,
):
    """Run a test query against the new index to verify retrieval quality."""
    print("\n" + "=" * 60)
    print(f'TEST QUERY: "{query}"')
    print("=" * 60)

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    for rank in range(top_k):
        meta       = results["metadatas"][0][rank]
        doc        = results["documents"][0][rank]
        similarity = 1.0 - results["distances"][0][rank]

        print(f"\nResult #{rank + 1} (Cosine Similarity: {similarity:.4f}):")
        print(f"  Product : {meta['product_category']} | Company: {meta['company']}")
        print(f"  Issue   : {meta['issue'][:80]}")
        print(f"  Excerpt : \"{doc[:200]}...\"")


def run_pipeline():
    sample_df = get_oversampled_stratified_sample()   # FIX 1 applied here
    chunks    = chunk_complaints(sample_df)
    client, collection, model = build_and_persist_vector_store(chunks)
    test_semantic_search(collection, model)            # Test with formerly-weak query


if __name__ == "__main__":
    run_pipeline()
