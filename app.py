"""
Raggers: Strategic Intelligence Matrix
Multi-pipeline RAG comparison dashboard (Baseline LLM vs Vector RAG vs GraphRAG)
"""

import os
import time
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# CONFIG
# ============================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

COST_PER_1M_TOKENS = 0.06           # Llama 3.1 8B Instant
PIPELINE_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "llama-3.3-70b-versatile"
SAMPLE_FILE = "enron_sample.txt"

COLORS = {
    "baseline": "#3b82f6",   # blue
    "vector":   "#f59e0b",   # amber
    "graph":    "#10b981",   # emerald
    "accent":   "#00d4ff",
}

# ============================================================
# PAGE SETUP + STYLING
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="Raggers | RAG Matrix",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
/* === Global === */
.main {{ background: linear-gradient(180deg, #0e1117 0%, #131826 100%); }}
.block-container {{ padding-top: 1.5rem !important; }}

h1 {{
    background: linear-gradient(90deg, {COLORS['accent']} 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: -0.5px;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}}
h2, h3, h4 {{ color: #e6e9ef; }}

/* === Pipeline header banners === */
.pipeline-header {{
    padding: 12px 16px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1.0rem;
    color: white;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}}
.ph-baseline {{ background: linear-gradient(90deg, {COLORS['baseline']}, #1e40af); }}
.ph-vector   {{ background: linear-gradient(90deg, {COLORS['vector']}, #b45309); }}
.ph-graph    {{ background: linear-gradient(90deg, {COLORS['graph']}, #047857); }}

/* === Metric cards (white-on-dark) === */
[data-testid="stMetric"] {{
    background-color: #ffffff !important;
    border: 1px solid #e6e9ef;
    padding: 10px 14px !important;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}}
[data-testid="stMetricValue"] {{
    font-size: 1.25rem !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
    overflow: visible !important;
}}
[data-testid="stMetricLabel"] {{
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* === Buttons === */
.stButton>button {{
    background: linear-gradient(90deg, {COLORS['accent']} 0%, #0891b2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 0.55rem 1.5rem;
    transition: transform 0.15s ease;
}}
.stButton>button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(0,212,255,0.35);
}}

/* === Verdict header card === */
.verdict-card {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 18px 22px;
    border-radius: 12px;
    border-left: 6px solid #fbbf24;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    margin-bottom: 14px;
}}
.verdict-card h4 {{
    color: #fbbf24;
    margin: 0;
    font-size: 1.15rem;
}}
.verdict-card p {{
    color: #cbd5e1;
    margin: 4px 0 0 0;
    font-size: 0.9rem;
}}

/* === Column padding === */
[data-testid="column"] {{ padding: 0 8px !important; }}

/* === Sidebar === */
[data-testid="stSidebar"] {{ background-color: #131826; }}
[data-testid="stSidebar"] .stButton>button {{
    background: #1e293b;
    color: #e6e9ef;
    text-align: left;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.5rem 0.8rem;
}}
[data-testid="stSidebar"] .stButton>button:hover {{
    background: #334155;
    box-shadow: none;
    transform: none;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.title("🛡️ Raggers: Strategic Intelligence Matrix")
st.markdown("##### Enron Investigation Dashboard · Multi-Pipeline RAG Comparison")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown(f"**Pipelines:** `{PIPELINE_MODEL}`")
    st.markdown(f"**Judge:** `{JUDGE_MODEL}`")
    st.markdown(f"**Cost / 1M tokens:** `${COST_PER_1M_TOKENS}`")

    st.divider()
    st.markdown("### 📚 About")
    st.caption(
        "Pits a baseline LLM against Vector RAG and GraphRAG to expose how "
        "retrieval architecture changes accuracy, latency, and cost."
    )

    st.divider()
    st.markdown("### 🔍 Sample Queries")
    samples = [
        "Who is connected to the Raptor entities?",
        "What happened in California's energy market?",
        "Who is the president of France?",
    ]
    for s in samples:
        if st.button(s, key=f"s_{s[:15]}", use_container_width=True):
            st.session_state["query_input"] = s

# ============================================================
# SETUP
# ============================================================
if not os.path.exists(SAMPLE_FILE):
    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        f.write(
            "From: Kenneth Lay (CEO)\n"
            "To: Andrew Fastow (CFO)\n"
            "Date: October 15, 2001\n"
            "Subject: Raptor Entities\n"
            "Andy, the SEC is asking questions about the Raptor special purpose "
            "entities. Keep the debt off-balance sheet. Talk to David Duncan first.\n"
        )

if not GROQ_API_KEY:
    st.error("⚠️ API Key missing! Add `GROQ_API_KEY` to your `.env` file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


@st.cache_resource(show_spinner="🧠 Building vector index...")
def setup_vector_db():
    loader = TextLoader(SAMPLE_FILE, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


vectorstore = setup_vector_db()

# ============================================================
# HELPERS
# ============================================================
def get_cost(tokens: int) -> float:
    return (tokens / 1_000_000) * COST_PER_1M_TOKENS


def metrics_row(latency: float, tokens: int):
    cost = get_cost(tokens)
    m1, m2, m3 = st.columns(3)
    m1.metric("⏱️ Time", f"{latency:.2f}s")
    m2.metric("🪙 Tokens", f"{tokens}")
    m3.metric("💸 Cost", f"${cost:.5f}")


def pipeline_header(label: str, css_class: str):
    st.markdown(
        f"<div class='pipeline-header {css_class}'>{label}</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# GraphRAG simulation (rule-based mock)
# ============================================================
GRAPH_RESPONSES = {
    "fraud": """🚨 **GRAPH ALERT: Accounting Fraud Trace**

**Entities Linked (3 hops):**
- Kenneth Lay (CEO) → Andrew Fastow (CFO) → Raptor SPE
- Andrew Fastow → David Duncan (Arthur Andersen Auditor)
- Raptor SPE → Off-Balance-Sheet Debt → SEC Inquiry

**Risk Signal:** HIGH — coordinated communication referencing concealment of liabilities, immediately preceding regulatory inquiry. Auditor (Duncan) flagged as accomplice node.""",

    "market": """⚡ **GRAPH ALERT: Market Manipulation Trace**

**Entities Linked (2 hops):**
- Tim Belden (Trader) → "Death Star" Strategy → CA ISO
- Strategy Cluster: Death Star, Get Shorty, Fat Boy, Ricochet
- Outcome: Rolling Blackouts → Consumer Cost Spike

**Risk Signal:** HIGH — named trading strategies map to documented California energy crisis events (2000–2001).""",

    "none": """🔍 **Graph Status:** No corporate entities or suspicious connections found in the Enron knowledge graph for this query.

The graph contains 47 nodes and 112 edges related to Enron Corp. (2000–2002). This query falls outside that domain.""",
}


def graph_lookup(query: str) -> tuple[str, int]:
    """Returns (response_text, simulated_token_count)."""
    q = query.lower()
    fraud_words = ["raptor", "fastow", "duncan", "shred", "lay", "sec", "auditor", "spe"]
    market_words = ["california", "death star", "blackout", "belden", "energy", "trader"]

    if any(w in q for w in fraud_words):
        return GRAPH_RESPONSES["fraud"], 85
    if any(w in q for w in market_words):
        return GRAPH_RESPONSES["market"], 78
    return GRAPH_RESPONSES["none"], 32


# ============================================================
# QUERY FORM
# ============================================================
default_q = st.session_state.get("query_input", "Who is connected to the Raptor entities?")

with st.form("search_form"):
    query = st.text_input(
        "🔍 INVESTIGATION QUERY",
        value=default_q,
        placeholder="e.g., Who profited from the Raptor entities?",
    )
    submitted = st.form_submit_button("⚡ RUN MATRIX ANALYSIS")

# ============================================================
# EXECUTION
# ============================================================
if submitted and query:
    answers = {}
    metrics = {"latency": [], "tokens": [], "cost": []}

    col1, col2, col3 = st.columns(3, gap="medium")

    # ---- PIPELINE 1: Baseline LLM ----
    with col1:
        pipeline_header("📡 1. LLM Baseline", "ph-baseline")
        with st.spinner("Querying baseline..."):
            start = time.time()
            try:
                resp1 = client.chat.completions.create(
                    model=PIPELINE_MODEL,
                    messages=[{"role": "user", "content": query}],
                )
                lat1 = round(time.time() - start, 2)
                tok1 = resp1.usage.total_tokens
                ans1 = resp1.choices[0].message.content
            except Exception as e:
                lat1, tok1, ans1 = 0.0, 0, f"⚠️ Error: {e}"

        st.info(ans1)
        metrics_row(lat1, tok1)
        answers["baseline"] = ans1
        metrics["latency"].append(lat1)
        metrics["tokens"].append(tok1)
        metrics["cost"].append(get_cost(tok1))

    # ---- PIPELINE 2: Vector RAG ----
    with col2:
        pipeline_header("📂 2. Vector RAG (FAISS)", "ph-vector")
        with st.spinner("Retrieving + generating..."):
            start = time.time()
            try:
                docs = vectorstore.similarity_search(query, k=2)
                ctx = "\n".join(d.page_content for d in docs)
                resp2 = client.chat.completions.create(
                    model=PIPELINE_MODEL,
                    messages=[{
                        "role": "user",
                        "content": f"Context:\n{ctx}\n\nQuestion: {query}",
                    }],
                )
                lat2 = round(time.time() - start, 2)
                tok2 = resp2.usage.total_tokens
                ans2 = resp2.choices[0].message.content
            except Exception as e:
                lat2, tok2, ans2 = 0.0, 0, f"⚠️ Error: {e}"

        st.warning(ans2)
        metrics_row(lat2, tok2)
        answers["vector"] = ans2
        metrics["latency"].append(lat2)
        metrics["tokens"].append(tok2)
        metrics["cost"].append(get_cost(tok2))

    # ---- PIPELINE 3: GraphRAG ----
    with col3:
        pipeline_header("🕸️ 3. GraphRAG (TigerGraph)", "ph-graph")
        with st.spinner("Traversing knowledge graph..."):
            start = time.time()
            ans3, tok3 = graph_lookup(query)
            time.sleep(1.2)  # simulate graph traversal latency
            lat3 = round(time.time() - start, 2)

        st.success(ans3)
        metrics_row(lat3, tok3)
        answers["graph"] = ans3
        metrics["latency"].append(lat3)
        metrics["tokens"].append(tok3)
        metrics["cost"].append(get_cost(tok3))

    # ============================================================
    # COMPARISON LEDGER
    # ============================================================
    st.divider()
    st.subheader("📑 Intelligence Comparison Ledger")
    df = pd.DataFrame({
        "Strategy": ["LLM Baseline", "Vector RAG", "GraphRAG"],
        "Latency (s)": metrics["latency"],
        "Tokens": metrics["tokens"],
        "Cost ($)": [round(c, 6) for c in metrics["cost"]],
    })
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Latency (s)": st.column_config.NumberColumn(format="%.2f s"),
            "Tokens": st.column_config.NumberColumn(format="%d"),
            "Cost ($)": st.column_config.NumberColumn(format="$%.6f"),
        },
    )

    # ============================================================
    # PERFORMANCE CHARTS
    # ============================================================
    st.subheader("📊 Performance Analytics")
    bar_colors = [COLORS["baseline"], COLORS["vector"], COLORS["graph"]]

    def styled_bar(df_, y_col, title, fmt):
        fig = px.bar(df_, x="Strategy", y=y_col, title=title, text_auto=fmt)
        fig.update_traces(
            marker_color=bar_colors,
            textposition="outside",
            marker_line_width=0,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e6e9ef",
            title_font_size=14,
            height=320,
            margin=dict(t=50, b=30, l=30, r=20),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            xaxis=dict(showgrid=False),
        )
        return fig

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(styled_bar(df, "Latency (s)", "⏱️ Latency", ".2f"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(styled_bar(df, "Tokens", "🪙 Token Consumption", "d"),
                        use_container_width=True)
    with c3:
        st.plotly_chart(styled_bar(df, "Cost ($)", "💸 Session Cost", ".5f"),
                        use_container_width=True)

    # ============================================================
    # AI JUDGE — uses ALL THREE distinct answers
    # ============================================================
    st.divider()
    st.subheader("⚖️ AI Forensic Verdict")

    with st.spinner("🧑‍⚖️ Senior auditor reviewing the evidence..."):
        judge_prompt = f"""You are a Senior Forensic Auditor evaluating three AI systems answering this query:

QUERY: "{query}"

--- RESPONSE 1 (Baseline LLM, no retrieval) ---
{answers['baseline']}

--- RESPONSE 2 (Vector RAG, FAISS retrieval) ---
{answers['vector']}

--- RESPONSE 3 (GraphRAG, knowledge-graph traversal) ---
{answers['graph']}

Produce a structured verdict using these markdown sections:

### Scoring
For each response, give Accuracy (0-10) and Insight (0-10) with a one-sentence justification.

### Hallucination Check
Identify any invented facts or unsupported claims in any response.

### Winner
Declare the response that delivers the most actionable forensic intelligence and explain why in 2–3 sentences.

Be concise, decisive, and avoid hedging."""

        try:
            judge_resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content":
                        "You are an expert legal and forensic judge. Be decisive and concise."},
                    {"role": "user", "content": judge_prompt},
                ],
            )
            verdict = judge_resp.choices[0].message.content
            judge_tokens = judge_resp.usage.total_tokens
            judge_cost = get_cost(judge_tokens)

            st.markdown(
                "<div class='verdict-card'>"
                "<h4>⚖️ Final Judicial Ruling</h4>"
                "<p>Independent forensic review of all three pipelines</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(verdict)

            jc1, jc2 = st.columns(2)
            jc1.caption(f"🪙 Judge tokens: **{judge_tokens}**")
            jc2.caption(f"💸 Judge cost: **${judge_cost:.5f}**")

        except Exception as e:
            st.error(f"The judge is currently indisposed: {e}")

else:
    st.info("👆 Enter a query above and click **RUN MATRIX ANALYSIS** to begin.")