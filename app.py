"""
app.py -- CrediTrust Complaint Intelligence Platform
Clean, human-centric UI inspired by modern workspace dashboards (MindMerge style).
Minimalist, high contrast, pure white chat input with high-contrast text, zero black hover artifacts.
"""

import os
import sys
import time
import datetime
import streamlit as st
import pandas as pd
import plotly.express as px

# Configure UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8")

# Ensure src directory is discoverable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from rag_pipeline import ComplaintRAG, MIN_SIMILARITY, MIN_SOURCES, GROQ_MODEL
from sql_engine import ComplaintSQL
from hybrid_router import HybridRouter
from digest_generator import WeeklyDigestGenerator

# -----------------------------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CrediTrust",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Clean, Minimalist MindMerge-Style Design System (Zero Black Hovering)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* Left Sidebar: Minimalist Dark Slate */
    [data-testid="stSidebar"] {
        background-color: #171821 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0;
    }

    /* Brand Header */
    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 16px;
    }
    .brand-icon {
        width: 32px;
        height: 32px;
        background: #FF6B35;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        color: #FFFFFF;
        font-size: 1rem;
    }
    .brand-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.01em;
    }

    /* Executive Digest Card in Sidebar */
    .digest-card {
        background: linear-gradient(135deg, #FF6B35 0%, #EA580C 100%);
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 14px;
        color: #FFFFFF !important;
    }
    .digest-card-title {
        font-size: 0.90rem;
        font-weight: 700;
        margin-bottom: 4px;
        color: #FFFFFF !important;
    }
    .digest-card-sub {
        font-size: 0.74rem;
        opacity: 0.9;
        line-height: 1.35;
        margin-bottom: 10px;
        color: #FFFFFF !important;
    }

    /* Top Workspace Header */
    .top-bar {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .top-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
    }
    .top-tag {
        font-size: 0.75rem;
        font-weight: 600;
        color: #059669;
        background: #ECFDF5;
        border: 1px solid #D1FAE5;
        padding: 3px 9px;
        border-radius: 9999px;
    }

    /* Response Card */
    .chat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .chat-card-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #F1F5F9;
        font-size: 0.76rem;
        font-weight: 600;
        color: #64748B;
    }

    /* User Message */
    .user-msg {
        background: #F1F5F9;
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 12px;
        font-size: 0.92rem;
        font-weight: 600;
        color: #0F172A;
    }

    /* Citation Box */
    .citation-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #FF6B35;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 8px;
        font-size: 0.82rem;
    }
    .citation-header {
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
    }
    .citation-text {
        color: #475569;
        line-height: 1.4;
        font-style: italic;
    }

    /* History Column */
    .history-head {
        font-size: 0.90rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #E2E8F0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* =========================================================================
       COMPLETE OVERRIDE: ELIMINATE HARSH BLACK HOVER STATES ACROSS ENTIRE UI
       ========================================================================= */
    /* Workspace Buttons: Soft warm hover instead of harsh black */
    .stMain button,
    div[data-testid="stMain"] button,
    div[data-testid="stButton"] > button {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }

    .stMain button:hover,
    div[data-testid="stMain"] button:hover,
    div[data-testid="stButton"] > button:hover {
        background-color: #FFF7ED !important;  /* Soft warm tint */
        color: #FF6B35 !important;            /* Orange text */
        border-color: #FF6B35 !important;      /* Soft orange border */
        box-shadow: 0 2px 8px rgba(255, 107, 53, 0.12) !important;
        transform: translateY(-1px) !important;
    }

    .stMain button:active,
    div[data-testid="stButton"] > button:active {
        background-color: #FFEDD5 !important;
        border-color: #EA580C !important;
        color: #EA580C !important;
        transform: translateY(0px) !important;
    }

    /* Sidebar Buttons: Soft subtle hover */
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] div[data-testid="stButton"] > button {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #F1F5F9 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="stSidebar"] button:hover,
    [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background-color: rgba(255, 107, 53, 0.15) !important;
        color: #FF8A50 !important;
        border-color: rgba(255, 107, 53, 0.4) !important;
        box-shadow: none !important;
    }

    /* Download and Primary Buttons */
    [data-testid="stDownloadButton"] > button {
        background-color: #FF6B35 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background-color: #EA580C !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(234, 88, 12, 0.25) !important;
    }

    /* Tabs: No black line/hover */
    button[data-baseweb="tab"] {
        color: #64748B !important;
        border-bottom: 2px solid transparent !important;
        transition: color 0.15s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #FF6B35 !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FF6B35 !important;
        border-bottom: 2px solid #FF6B35 !important;
        background: transparent !important;
    }

    /* =========================================================================
       PURE WHITE CHAT INPUT CONTAINER & CRISP BLACK TEXT VISIBILITY
       ========================================================================= */
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"],
    footer {
        background-color: transparent !important;
    }

    [data-testid="stChatInputContainer"],
    [data-testid="stChatInputContainer"] > div {
        background-color: transparent !important;
    }

    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] form {
        background-color: #FFFFFF !important;
        border: 2px solid #94A3B8 !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
        padding: 4px 10px !important;
        margin-bottom: 8px !important;
    }

    [data-testid="stChatInput"]:focus-within,
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #FF6B35 !important;
        box-shadow: 0 4px 24px rgba(255, 107, 53, 0.25) !important;
    }

    /* Force the textarea to pure white background and crisp deep black text */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"],
    textarea[data-testid="stChatInputTextArea"],
    div[data-testid="stChatInput"] textarea {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        caret-color: #FF6B35 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInputTextArea"]::placeholder,
    textarea[data-testid="stChatInputTextArea"]::placeholder {
        color: #64748B !important;
        -webkit-text-fill-color: #64748B !important;
        font-weight: 400 !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #FF6B35 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        border: none !important;
    }
    [data-testid="stChatInput"] button:hover {
        background-color: #EA580C !important;
    }
    [data-testid="stChatInput"] button svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Smooth Scroller Script
# -----------------------------------------------------------------------------
def render_auto_scroller():
    scroller_html = """
    <div id="latest-response-anchor" style="height: 1px; width: 100%; margin-top: 10px;"></div>
    <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
         style="display:none;"
         onload="(function() {
            function forceScrollBottom() {
                var targets = [
                    document.querySelector('.stMain'),
                    document.querySelector('section.main'),
                    document.querySelector('[data-testid=\\'stAppViewContainer\\']'),
                    document.documentElement,
                    document.body
                ];
                targets.forEach(function(el) {
                    if (el) {
                        try { el.scrollTo({ top: el.scrollHeight + 10000, behavior: 'smooth' }); } catch(e){}
                    }
                });
                var anchor = document.getElementById('latest-response-anchor');
                if (anchor) {
                    try { anchor.scrollIntoView({ behavior: 'smooth', block: 'end' }); } catch(e){}
                }
            }
            forceScrollBottom();
            setTimeout(forceScrollBottom, 150);
            setTimeout(forceScrollBottom, 600);
         })()" />
    """
    st.markdown(scroller_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Singleton Initializer
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_hybrid_system():
    rag_engine = ComplaintRAG()
    sql_engine = ComplaintSQL()
    router = HybridRouter(rag_engine=rag_engine, sql_engine=sql_engine)
    digest_gen = WeeklyDigestGenerator()
    return rag_engine, sql_engine, router, digest_gen


try:
    with st.spinner("Connecting..."):
        rag, sql, router, digest_gen = load_hybrid_system()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_history" not in st.session_state:
    st.session_state.query_history = [
        "Which 5 companies receive the most Credit Card complaints?",
        "Why do people complain about money transfers?",
        "What complaints do borrowers file regarding loan interest?"
    ]
if "show_digest" not in st.session_state:
    st.session_state.show_digest = False
if "digest_memo" not in st.session_state:
    st.session_state.digest_memo = None
if "digest_filepath" not in st.session_state:
    st.session_state.digest_filepath = None

has_messages = len(st.session_state.messages) > 0


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
        <div class="brand-icon">C</div>
        <div class="brand-name">CrediTrust</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Engine Mode**")
    engine_mode = st.radio(
        "Mode:",
        options=["Hybrid Auto-Router", "Qualitative RAG Only", "SQL Analytics Only"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("**Product Filter**")
    product_options = [
        "All Products",
        "Credit Card",
        "Personal Loan",
        "Savings Account",
        "Money Transfer"
    ]
    selected_product = st.selectbox(
        "Product:",
        options=product_options,
        index=0,
        label_visibility="collapsed"
    )
    product_filter = None if selected_product == "All Products" else selected_product

    # Executive Digest Box
    st.markdown("""
    <div class="digest-card">
        <div class="digest-card-title">Executive Digest</div>
        <div class="digest-card-sub">Export compliance summary and top risks.</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Generate Weekly Digest", use_container_width=True):
        with st.spinner("Generating digest..."):
            memo_text = digest_gen.generate_digest()
            today_str = datetime.date.today().strftime("%Y_%m_%d")
            out_filename = f"creditrust_compliance_digest_{today_str}.md"
            out_filepath = os.path.join(REPORTS_DIR, out_filename)

            with open(out_filepath, "w", encoding="utf-8") as f:
                f.write(memo_text)

            st.session_state.digest_memo = memo_text
            st.session_state.digest_filepath = out_filepath
            st.session_state.show_digest = True

    if st.session_state.digest_memo:
        st.download_button(
            label="📥 Download Digest (.md)",
            data=st.session_state.digest_memo,
            file_name=os.path.basename(st.session_state.digest_filepath),
            mime="text/markdown",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.show_digest = False
        st.session_state.digest_memo = None
        st.rerun()


# -----------------------------------------------------------------------------
# Main Content Area: 2 Columns (Workspace 75% + History 25%)
# -----------------------------------------------------------------------------
col_main, col_hist = st.columns([3.2, 1.0], gap="medium")

selected_query = None

# History Drawer (Right Column)
with col_hist:
    st.markdown(f"""
    <div class="history-head">
        <span>History</span>
        <span style="font-size: 0.75rem; color: #64748B;">{len(st.session_state.query_history)}</span>
    </div>
    """, unsafe_allow_html=True)

    for idx, item in enumerate(st.session_state.query_history):
        label = item if len(item) <= 36 else f"{item[:34]}..."
        if st.button(f"• {label}", key=f"hist_{idx}", use_container_width=True):
            selected_query = item

    if st.button("Clear History", key="btn_clear_hist", use_container_width=True):
        st.session_state.query_history = []
        st.rerun()

# Workspace (Center Column)
with col_main:
    st.markdown("""
    <div class="top-bar">
        <h3 class="top-title">Complaints Copilot</h3>
        <span class="top-tag">CFPB Database Active</span>
    </div>
    """, unsafe_allow_html=True)

    # Executive Digest View
    if st.session_state.show_digest and st.session_state.digest_memo:
        with st.expander("📑 Weekly Executive Digest", expanded=True):
            st.markdown(st.session_state.digest_memo)
            st.download_button(
                label="📥 Download (.md)",
                data=st.session_state.digest_memo,
                file_name=os.path.basename(st.session_state.digest_filepath),
                mime="text/markdown"
            )

    # Quick Prompts (Only shown on empty conversation)
    if not has_messages and not st.session_state.show_digest:
        st.caption("Suggested Questions:")
        p1, p2, p3 = st.columns(3)
        with p1:
            if st.button("Top Credit Card Issuers", use_container_width=True):
                selected_query = "Which 5 companies receive the most Credit Card complaints?"
        with p2:
            if st.button("Money Transfer Delays", use_container_width=True):
                selected_query = "Why do people complain about money transfers?"
        with p3:
            if st.button("Personal Loan Interest", use_container_width=True):
                selected_query = "What complaints do borrowers file regarding loan interest?"

    # Message Feed
    for m_idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            intent = msg.get("intent", "RAG")
            source_count = len(msg.get("sources", []))

            st.markdown(f"""
            <div class="chat-card">
                <div class="chat-card-meta">
                    <span>{ '📊 Quantitative SQL' if intent == 'SQL' else '🔍 Qualitative RAG' }</span>
                    <span>{f'{source_count} Citations' if intent == 'RAG' else 'Database Query'}</span>
                </div>
            """, unsafe_allow_html=True)

            if intent == "SQL":
                tab_ans, tab_chart, tab_data = st.tabs(["Answer", "Chart", "Data Table"])
                with tab_ans:
                    st.markdown(msg["content"])
                with tab_chart:
                    df = msg.get("data")
                    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
                        if num_cols and cat_cols:
                            fig = px.bar(
                                df,
                                x=cat_cols[0],
                                y=num_cols[0],
                                title=f"Distribution by {cat_cols[0].replace('_', ' ').title()}",
                                color_discrete_sequence=["#FF6B35"],
                                template="plotly_white"
                            )
                            fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No chart available.")
                with tab_data:
                    if msg.get("data") is not None and isinstance(msg["data"], pd.DataFrame):
                        st.dataframe(msg["data"], use_container_width=True)
            else:
                tab_ans, tab_sources = st.tabs(["Answer", "Sources"])
                with tab_ans:
                    st.markdown(msg["content"])
                with tab_sources:
                    sources = msg.get("sources", [])
                    if sources:
                        for src in sources:
                            meta = src.get("metadata", {})
                            sim = src.get("similarity", 0.0)
                            st.markdown(f"""
                            <div class="citation-item">
                                <div class="citation-header">
                                    <span>{meta.get('company', 'Unknown')} · #{meta.get('complaint_id')}</span>
                                    <span>Similarity: {sim:.2f}</span>
                                </div>
                                <div class="citation-text">"{src.get('text', '')}"</div>
                            </div>
                            """, unsafe_allow_html=True)
                    elif msg.get("guard_active"):
                        st.info("No matching complaints found for this topic.")

            # Evaluation Feedback Buttons
            fb1, fb2, _ = st.columns([1, 1, 6])
            with fb1:
                if st.button("👍 Helpful", key=f"up_{m_idx}"):
                    st.toast("Recorded: Helpful")
            with fb2:
                if st.button("👎 Unhelpful", key=f"down_{m_idx}"):
                    st.toast("Recorded: Unhelpful")

            st.markdown("</div>", unsafe_allow_html=True)

    # High Visibility Pure White Chat Input
    user_input = st.chat_input("Ask a question about complaints or trends...")

    active_query = None
    if selected_query:
        active_query = selected_query
    elif user_input:
        active_query = user_input

    if active_query:
        # Add to history
        if active_query not in st.session_state.query_history:
            st.session_state.query_history.insert(0, active_query)
            st.session_state.query_history = st.session_state.query_history[:15]

        # Append User Message
        st.session_state.messages.append({"role": "user", "content": active_query})

        # Process Query
        start_t = time.time()
        if engine_mode == "SQL Analytics Only":
            active_intent = "SQL"
        elif engine_mode == "Qualitative RAG Only":
            active_intent = "RAG"
        else:
            active_intent = router.classify_intent(active_query)

        with st.spinner("Retrieving answer..."):
            if active_intent == "SQL":
                sql_res = sql.query_and_analyze(active_query)
                answer = sql_res["synthesis"]
                sql_query = sql_res["sql_query"]
                data = sql_res["data"]
                sources = []
                guard_active = False
            else:
                rag_res = rag.generate_response(
                    query=active_query,
                    top_k=6,
                    product_filter=product_filter,
                    min_similarity=0.50
                )
                answer = rag_res["answer"]
                sql_query = None
                data = None
                sources = rag_res["sources"]
                guard_active = len(sources) < MIN_SOURCES

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "intent": active_intent,
            "sql_query": sql_query,
            "data": data,
            "sources": sources,
            "guard_active": guard_active
        })
        st.rerun()

if has_messages or active_query:
    render_auto_scroller()
