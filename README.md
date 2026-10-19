# Intelligent Complaint Analysis for Financial Services
### Building a RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights

**Company:** CrediTrust Financial  
**Objective:** Transform unstructured customer complaints into an internal AI decision-support tool using Retrieval-Augmented Generation (RAG).  
**Target Financial Products:**
* Credit Cards
* Personal Loans
* Savings Accounts
* Money Transfers

---

## 🏗️ Project Architecture & Roadmap

1. **Task 1: Exploratory Data Analysis & Preprocessing**
   * Filter CFPB complaint dataset to 4 core financial services.
   * Clean text narratives (boilerplates, special characters, normalization).
2. **Task 2: Text Chunking, Embedding & Vector Store Indexing**
   * Implement character chunking with overlap.
   * Generate vector embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
   * Persist embeddings with metadata into ChromaDB / FAISS.
3. **Task 3: RAG Core Engine & Evaluation**
   * Semantic retrieval with top-k similarity search.
   * Grounded prompt engineering.
   * Qualitative evaluation across representative business queries.
4. **Task 4: Interactive Web Interface**
   * Streamlit / Gradio interactive chat with transparent source citations.
