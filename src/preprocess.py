"""
preprocess.py -- Stream, Filter, and Clean CFPB Complaints for CrediTrust RAG
Filters the raw 6.0 GB CFPB dataset down to 4 target financial products:
1. Credit Cards
2. Personal Loans
3. Savings Accounts
4. Money Transfers
Cleans consumer narratives (stripping boilerplate, redactions, URLs, noise).
Saves output to data/filtered_complaints.csv
"""

import os
import re
import pandas as pd

# Mappings from raw CFPB product names to CrediTrust standard categories
PRODUCT_MAPPING = {
    # 1. Credit Cards
    "Credit card": "Credit Card",
    "Credit card or prepaid card": "Credit Card",
    "Prepaid card": "Credit Card",

    # 2. Personal Loans
    "Payday loan, title loan, personal loan, or advance loan": "Personal Loan",
    "Consumer Loan": "Personal Loan",
    "Payday loan": "Personal Loan",

    # 3. Savings Accounts
    "Checking or savings account": "Savings Account",
    "Bank account or service": "Savings Account",

    # 4. Money Transfers
    "Money transfer, virtual currency, or money service": "Money Transfer",
    "Money transfers": "Money Transfer",
}

# Default raw dataset path in Downloads
DEFAULT_RAW_PATH = r"C:\Users\hp\Downloads\complaints.csv\complaints.csv"
OUTPUT_PATH = os.path.join("data", "filtered_complaints.csv")


def clean_narrative(text: str) -> str:
    """Clean and normalize a raw CFPB consumer complaint narrative."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs if any
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove privacy redactions (e.g. XXXX, XX/XX/XXXX, XXXXXXXX)
    text = re.sub(r"[xX]{2,}(/[xX]{2,})*", "", text)

    # Remove common boilerplate phrases
    text = re.sub(r"i am writing (to file )?a complaint\w*", "", text)
    text = re.sub(r"to whom it may concern\w*", "", text)
    text = re.sub(r"please be advised that\w*", "", text)

    # Remove non-informative symbols while keeping basic punctuation
    text = re.sub(r"[^a-zA-Z0-9.,!?$\s]", " ", text)

    # Collapse multiple spaces and linebreaks
    text = re.sub(r"\s+", " ", text).strip()

    return text


def stream_and_filter_complaints(
    raw_path: str = DEFAULT_RAW_PATH,
    output_path: str = OUTPUT_PATH,
    chunk_size: int = 100_000,
    max_records: int = None,
) -> str:
    """
    Stream large CFPB CSV in chunks to avoid memory issues.
    Filters target products with non-empty narratives, cleans text, and saves to CSV.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw CFPB dataset not found at: {raw_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Starting chunked stream from: {raw_path}")
    print(f"Target products: {sorted(set(PRODUCT_MAPPING.values()))}")
    print(f"Output destination: {output_path}\n")

    first_chunk = True
    total_scanned = 0
    total_saved = 0
    product_counts = {p: 0 for p in set(PRODUCT_MAPPING.values())}

    # Stream chunks of 100,000 rows
    for chunk_idx, chunk in enumerate(pd.read_csv(raw_path, chunksize=chunk_size, low_memory=False)):
        total_scanned += len(chunk)

        # 1. Filter rows where Product is in our target mapping
        chunk = chunk[chunk["Product"].isin(PRODUCT_MAPPING.keys())].copy()

        # 2. Filter rows with non-empty Consumer complaint narrative
        chunk = chunk[chunk["Consumer complaint narrative"].notna()].copy()

        if chunk.empty:
            print(f"  Chunk {chunk_idx + 1:>3}: Scanned {total_scanned:>10,} rows | 0 target complaints in this chunk")
            continue

        # 3. Map to CrediTrust standard product categories
        chunk["product_category"] = chunk["Product"].map(PRODUCT_MAPPING)

        # 4. Clean the narrative text
        chunk["cleaned_narrative"] = chunk["Consumer complaint narrative"].astype(str).apply(clean_narrative)

        # Filter out narratives that became too short after cleaning (< 20 characters)
        chunk = chunk[chunk["cleaned_narrative"].str.len() >= 20].copy()

        # 5. Select and standardize columns for RAG
        cols_to_keep = {
            "Complaint ID": "complaint_id",
            "Date received": "date_received",
            "product_category": "product_category",
            "Product": "product_raw",
            "Issue": "issue",
            "Sub-issue": "sub_issue",
            "Company": "company",
            "State": "state",
            "cleaned_narrative": "cleaned_narrative",
        }

        # Keep existing columns that match
        available_cols = [c for c in cols_to_keep.keys() if c in chunk.columns]
        chunk_clean = chunk[available_cols].rename(columns=cols_to_keep)

        # Update product counts
        for p, count in chunk_clean["product_category"].value_counts().items():
            product_counts[p] += count

        # 6. Append to output CSV file
        mode = "w" if first_chunk else "a"
        chunk_clean.to_csv(output_path, mode=mode, index=False, header=first_chunk, encoding="utf-8")
        first_chunk = False

        total_saved += len(chunk_clean)
        print(f"  Chunk {chunk_idx + 1:>3}: Scanned {total_scanned:>10,} rows | Saved {len(chunk_clean):>6,} complaints (Total: {total_saved:,})")

        if max_records and total_saved >= max_records:
            print(f"\nReached target limit of {max_records:,} records.")
            break

    print("\n" + "=" * 55)
    print("        STREAMING & PREPROCESSING COMPLETE")
    print("=" * 55)
    print(f"Total raw rows scanned:    {total_scanned:,}")
    print(f"Total target complaints:   {total_saved:,}")
    print(f"Saved file size:           {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print("\nComplaints by Product Category:")
    for p, c in product_counts.items():
        pct = (c / total_saved) * 100 if total_saved else 0
        print(f"  - {p:<20}: {c:>8,} ({pct:.1f}%)")

    return output_path


if __name__ == "__main__":
    stream_and_filter_complaints()
