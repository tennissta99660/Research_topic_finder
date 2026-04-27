# app.py — Streamlit UI for CARTOGRAPH v2
import config  # MUST be first — sets env vars to suppress transformers warnings
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

matplotlib.use("Agg")

st.set_page_config(
    page_title="CARTOGRAPH — Research Navigator",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (
    ARXIV_MAX_RESULTS, TOP_K_OUTPUT, NUM_GLOBAL_DIMENSIONS,
    DEFAULT_TOPIC_DIMENSIONS, MAX_TOPIC_DIMENSIONS, MIN_TOPIC_DIMENSIONS,
)
import networkx as nx
from db import get_or_create_topic, get_papers_by_topic, get_unscored_papers, get_unembedded_papers
from dimensions import get_all_dimensions, get_global_dimensions, get_topic_dimensions, is_global_dimension, _cache_path
from ingest import fetch_and_store_papers
from score import score_papers
from embed import embed_papers
from graph import build_all_edges, load_combined_graph, load_graph
from query import translate_query, retrieve_with_expansion, retrieve_on_combined_graph
from gap import detect_gap, synthesize_directions

logging.basicConfig(level=logging.INFO)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Keyframes ──────────────────────────────────────────────────────────── */
@keyframes aurora {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 8px rgba(139, 92, 246, 0.0); }
    50%      { box-shadow: 0 0 20px rgba(139, 92, 246, 0.15); }
}
@keyframes borderGlow {
    0%, 100% { border-color: rgba(139, 92, 246, 0.15); }
    50%      { border-color: rgba(99, 202, 255, 0.35); }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Base ───────────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #06060b 0%, #0c0c1d 25%, #0f1628 50%, #0a0f1e 75%, #080812 100%);
    background-size: 400% 400%;
    animation: aurora 30s ease infinite;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,12,20,0.97) 0%, rgba(15,18,30,0.97) 100%);
    border-right: 1px solid rgba(139, 92, 246, 0.12);
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    background: linear-gradient(135deg, #a78bfa, #63caff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* ── Paper Cards ────────────────────────────────────────────────────────── */
.paper-card {
    background: linear-gradient(135deg, rgba(15,18,30,0.9), rgba(10,12,22,0.85));
    border: 1px solid rgba(139, 92, 246, 0.12);
    border-radius: 20px; padding: 28px; margin-bottom: 20px;
    backdrop-filter: blur(16px);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeUp 0.5s ease both;
    position: relative;
    overflow: hidden;
}
.paper-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.5), rgba(99, 202, 255, 0.5), transparent);
    opacity: 0;
    transition: opacity 0.4s ease;
}
.paper-card:hover {
    border-color: rgba(139, 92, 246, 0.35);
    box-shadow: 0 12px 40px rgba(139, 92, 246, 0.08), 0 4px 16px rgba(99, 202, 255, 0.05);
    transform: translateY(-3px);
}
.paper-card:hover::before { opacity: 1; }
.paper-card h4 {
    color: #ede9fe; margin: 0 0 10px 0; font-weight: 700; font-size: 1.08rem;
    letter-spacing: -0.3px; line-height: 1.4;
}
.paper-card .distance-badge {
    display: inline-block;
    background: linear-gradient(135deg, #8b5cf6, #6d28d9);
    color: white; padding: 4px 14px; border-radius: 20px;
    font-size: 0.76rem; font-weight: 600; margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 12px rgba(139, 92, 246, 0.25);
}
.paper-card .paper-link {
    color: #a78bfa; text-decoration: none; font-size: 0.85rem;
    transition: color 0.2s ease; font-weight: 500;
}
.paper-card .paper-link:hover { color: #c4b5fd; }

/* ── Direction Cards ────────────────────────────────────────────────────── */
.direction-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.06), rgba(10, 12, 22, 0.92));
    border: 1px solid rgba(52, 211, 153, 0.2); border-radius: 20px;
    padding: 28px; margin-bottom: 20px;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    animation: fadeUp 0.5s ease both;
}
.direction-card:hover {
    border-color: rgba(52, 211, 153, 0.4);
    box-shadow: 0 8px 32px rgba(52, 211, 153, 0.06);
}
.direction-card h4 {
    background: linear-gradient(135deg, #34d399, #6ee7b7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 14px 0; font-weight: 800; font-size: 1.05rem;
}
.direction-card p { color: #d1d5db; line-height: 1.7; margin: 8px 0; font-size: 0.92rem; }
.direction-card .experiment-box {
    background: rgba(139, 92, 246, 0.06);
    border-left: 3px solid #a78bfa;
    padding: 14px 18px; border-radius: 0 12px 12px 0; margin-top: 14px;
    font-size: 0.88rem; color: #9ca3af;
}

/* ── Gap Pills ──────────────────────────────────────────────────────────── */
.gap-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(245, 158, 11, 0.04));
    border: 1px solid rgba(245, 158, 11, 0.3); color: #fbbf24;
    padding: 7px 16px; border-radius: 24px; font-size: 0.8rem;
    font-weight: 500; margin: 4px 4px 4px 0;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.2px;
    transition: all 0.3s ease;
}
.gap-pill:hover { background: rgba(245, 158, 11, 0.18); transform: scale(1.02); }
.gap-pill-global {
    display: inline-block;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(239, 68, 68, 0.04));
    border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171;
    padding: 7px 16px; border-radius: 24px; font-size: 0.8rem;
    font-weight: 500; margin: 4px 4px 4px 0;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.2px;
    transition: all 0.3s ease;
}
.gap-pill-global:hover { background: rgba(239, 68, 68, 0.18); transform: scale(1.02); }

/* ── Dimension Chips ────────────────────────────────────────────────────── */
.dim-chip-global {
    display: inline-block;
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.1), rgba(16, 185, 129, 0.05));
    border: 1px solid rgba(52, 211, 153, 0.25); color: #6ee7b7;
    padding: 6px 14px; border-radius: 14px; font-size: 0.8rem; margin: 3px;
    font-weight: 600; transition: all 0.3s ease; cursor: default;
}
.dim-chip-global:hover {
    background: rgba(52, 211, 153, 0.18);
    box-shadow: 0 2px 12px rgba(52, 211, 153, 0.12);
}
.dim-chip-topic {
    display: inline-block;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(99, 202, 255, 0.05));
    border: 1px solid rgba(139, 92, 246, 0.2); color: #a78bfa;
    padding: 6px 14px; border-radius: 14px; font-size: 0.8rem; margin: 3px;
    font-weight: 500; transition: all 0.3s ease; cursor: default;
}
.dim-chip-topic:hover {
    background: rgba(139, 92, 246, 0.15);
    box-shadow: 0 2px 12px rgba(139, 92, 246, 0.1);
}

/* ── Stat Boxes ─────────────────────────────────────────────────────────── */
.stat-box {
    background: linear-gradient(135deg, rgba(15, 18, 30, 0.9), rgba(10, 12, 22, 0.85));
    border: 1px solid rgba(139, 92, 246, 0.1); border-radius: 16px;
    padding: 22px; text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    animation: pulseGlow 4s ease-in-out infinite;
}
.stat-box:hover {
    border-color: rgba(139, 92, 246, 0.3);
    transform: translateY(-2px);
}
.stat-box .stat-value {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #63caff, #34d399);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: gradientShift 4s ease infinite;
}
.stat-box .stat-label {
    color: #6b7280; font-size: 0.78rem; font-weight: 600; margin-top: 6px;
    text-transform: uppercase; letter-spacing: 1px;
}

/* ── Hero ───────────────────────────────────────────────────────────────── */
.hero-title {
    font-size: 3.2rem; font-weight: 900; letter-spacing: -2px; margin-bottom: 6px;
    background: linear-gradient(135deg, #a78bfa 0%, #63caff 30%, #34d399 60%, #fbbf24 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
}
.hero-subtitle {
    color: #6b7280; font-size: 1.05rem; margin-bottom: 32px; line-height: 1.6;
    font-weight: 400;
}

/* ── Legends ────────────────────────────────────────────────────────────── */
.latent-legend {
    display: inline-block; padding: 5px 14px; border-radius: 14px;
    font-size: 0.75rem; font-weight: 600; margin: 3px 5px;
    transition: all 0.3s ease; cursor: default;
    font-family: 'JetBrains Mono', monospace;
}
.latent-legend:hover { transform: translateY(-1px); }
.latent-legend-global {
    background: linear-gradient(135deg, rgba(52,211,153,0.12), rgba(16,185,129,0.06));
    color: #6ee7b7; border: 1px solid rgba(52,211,153,0.25);
}
.latent-legend-topic {
    background: linear-gradient(135deg, rgba(139,92,246,0.1), rgba(99,202,255,0.06));
    color: #a78bfa; border: 1px solid rgba(139,92,246,0.2);
}
.latent-legend-desired {
    background: linear-gradient(135deg, rgba(251,191,36,0.12), rgba(245,158,11,0.06));
    color: #fbbf24; border: 1px solid rgba(251,191,36,0.25);
}

/* ── Section Dividers ───────────────────────────────────────────────────── */
.section-header {
    display: flex; align-items: center; gap: 12px; margin: 32px 0 18px 0;
}
.section-header .section-icon {
    width: 40px; height: 40px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(99,202,255,0.08));
    border: 1px solid rgba(139,92,246,0.2);
}
.section-header .section-text {
    font-size: 1.3rem; font-weight: 800; color: #e5e7eb; letter-spacing: -0.5px;
}

/* ── Onboarding Card ────────────────────────────────────────────────────── */
.onboard-card {
    background: linear-gradient(135deg, rgba(15,18,30,0.85), rgba(10,12,22,0.8));
    border: 1px solid rgba(139,92,246,0.15); border-radius: 24px;
    padding: 40px; margin-top: 20px;
    backdrop-filter: blur(16px);
    animation: fadeUp 0.6s ease both, borderGlow 6s ease-in-out infinite;
}
.onboard-card h3 {
    background: linear-gradient(135deg, #a78bfa, #63caff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800; font-size: 1.4rem; margin-bottom: 20px;
}
.onboard-step {
    display: flex; align-items: flex-start; gap: 14px; margin: 16px 0;
    padding: 14px 18px; border-radius: 14px;
    background: rgba(139, 92, 246, 0.04);
    border: 1px solid rgba(139, 92, 246, 0.08);
    transition: all 0.3s ease;
}
.onboard-step:hover {
    background: rgba(139, 92, 246, 0.08);
    border-color: rgba(139, 92, 246, 0.18);
    transform: translateX(4px);
}
.onboard-step .step-num {
    min-width: 32px; height: 32px; border-radius: 10px;
    background: linear-gradient(135deg, #8b5cf6, #6d28d9);
    color: white; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
}
.onboard-step .step-content { color: #d1d5db; font-size: 0.92rem; line-height: 1.5; }
.onboard-step .step-content strong { color: #e5e7eb; }

/* ── Streamlit Overrides ────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #8b5cf6, #63caff, #34d399);
    background-size: 200% auto;
    animation: gradientShift 2s ease infinite;
}
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #8b5cf6, #63caff) !important;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(139, 92, 246, 0.1) !important;
    border-radius: 16px !important;
    background: rgba(10, 12, 22, 0.5) !important;
    transition: border-color 0.3s ease;
}
div[data-testid="stExpander"]:hover {
    border-color: rgba(139, 92, 246, 0.25) !important;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(15,18,30,0.8), rgba(10,12,22,0.7));
    border: 1px solid rgba(139, 92, 246, 0.08);
    border-radius: 14px; padding: 12px;
}
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Radar Chart Helper ────────────────────────────────────────────────────────
def render_radar_chart(score_vector, desired_vector, dim_names, title=""):
    # For large dim counts, show only top-15 most relevant
    if len(dim_names) > 15:
        indices = list(range(min(4, len(dim_names))))  # always show globals
        topic_indices = list(range(4, len(dim_names)))
        diffs = [(abs(score_vector[i] - (desired_vector[i] if desired_vector[i] != -1 else 0.5)), i)
                 for i in topic_indices]
        diffs.sort(reverse=True)
        indices += [idx for _, idx in diffs[:11]]
        indices.sort()
        dim_names = [dim_names[i] for i in indices]
        score_vector = [score_vector[i] for i in indices]
        desired_vector = [desired_vector[i] for i in indices]

    n = len(dim_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    paper_vals = score_vector + score_vector[:1]
    desired_vals = [(v if v != -1.0 else 0.5) for v in desired_vector] + \
                   [desired_vector[0] if desired_vector[0] != -1.0 else 0.5]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#06060b")
    ax.set_facecolor("#06060b")
    ax.fill(angles, paper_vals, alpha=0.15, color="#a78bfa")
    ax.plot(angles, paper_vals, linewidth=2.5, color="#a78bfa", label="Paper")
    ax.fill(angles, desired_vals, alpha=0.08, color="#34d399")
    ax.plot(angles, desired_vals, linewidth=2, color="#34d399", linestyle="--", label="Query")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_names, fontsize=7, color="#9ca3af")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="#374151")
    ax.spines["polar"].set_color("#1f2937")
    ax.grid(color="#1f2937", linewidth=0.5)
    if title:
        ax.set_title(title, fontsize=10, color="#e5e7eb", pad=20, fontweight=600)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9,
              facecolor="#0f1628", edgecolor="#1f2937", labelcolor="#d1d5db")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🗺️ CARTOGRAPH v2")
    st.markdown("*Multi-layer research navigator*")
    st.markdown("---")

    st.markdown("### 📌 Topic")
    topic = st.text_input("Research field", placeholder="e.g. Graph Neural Networks", label_visibility="collapsed")

    n_topic_dims = st.slider(
        "Topic-specific dimensions",
        min_value=MIN_TOPIC_DIMENSIONS, max_value=MAX_TOPIC_DIMENSIONS,
        value=DEFAULT_TOPIC_DIMENSIONS,
        help=f"Plus {NUM_GLOBAL_DIMENSIONS} global dims = total dimensions",
    )
    st.caption(f"Total dimensions: {NUM_GLOBAL_DIMENSIONS} global + {n_topic_dims} topic = **{NUM_GLOBAL_DIMENSIONS + n_topic_dims}**")

    if topic:
        topic_id = get_or_create_topic(topic, n_topic_dims)
        all_papers = get_papers_by_topic(topic_id)
        unscored = get_unscored_papers(topic_id)
        unembedded = get_unembedded_papers(topic_id)
        scored_count = len(all_papers) - len(unscored)
        embedded_count = len(all_papers) - len(unembedded)

        st.markdown("---")
        st.markdown("### 📊 Status")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Papers", len(all_papers))
            st.metric("Scored", scored_count)
        with c2:
            st.metric("Embedded", embedded_count)
            st.metric("Unscored", len(unscored))

        st.markdown("---")
        st.markdown("### ⚙️ Pipeline")

        if st.button("📥 Fetch Papers", use_container_width=True):
            with st.spinner(f"Fetching up to {ARXIV_MAX_RESULTS} papers..."):
                try:
                    n = fetch_and_store_papers(topic, topic_id)
                    st.success(f"Fetched {n} papers!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

        if len(unscored) > 0 and st.button(f"📝 Score {len(unscored)} Papers", use_container_width=True):
            with st.spinner("Scoring papers with LLM..."):
                try:
                    dims = get_all_dimensions(topic, n_topic_dims)
                    n = score_papers(topic_id, dims)
                    st.success(f"Scored {n} papers!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Scoring failed: {e}")

        if len(unembedded) > 0 and st.button(f"🧬 Embed {len(unembedded)} Papers", use_container_width=True):
            with st.spinner("Computing SBERT embeddings..."):
                try:
                    n = embed_papers(topic_id)
                    st.success(f"Embedded {n} papers!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Embedding failed: {e}")

        if embedded_count > 1 and scored_count > 1:
            if st.button("🔗 Build All Edges", use_container_width=True):
                with st.spinner("Building semantic + dimension edges..."):
                    try:
                        dims = get_all_dimensions(topic, n_topic_dims)
                        result = build_all_edges(topic_id, dims)
                        st.success(f"Semantic: {result['semantic_edges']}, Dimension: {result['dimension_edges']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Edge building failed: {e}")

        st.markdown("---")
        if st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary"):
            progress = st.progress(0, text="Starting...")
            try:
                progress.progress(5, text="Generating dimensions...")
                dims = get_all_dimensions(topic, n_topic_dims)
                if len(all_papers) == 0:
                    progress.progress(15, text="Fetching papers...")
                    fetch_and_store_papers(topic, topic_id)
                progress.progress(35, text="Scoring papers...")
                if get_unscored_papers(topic_id):
                    score_papers(topic_id, dims)
                progress.progress(60, text="Embedding papers...")
                if get_unembedded_papers(topic_id):
                    embed_papers(topic_id)
                progress.progress(80, text="Building multi-layer graph...")
                build_all_edges(topic_id, dims)
                progress.progress(100, text="Done!")
                st.success("✅ Pipeline complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

        # Dimension viewer — only show if already cached (don't trigger LLM)
        st.markdown("---")
        st.markdown("### 🧭 Dimensions")
        import os
        if os.path.exists(_cache_path(topic)):
            dims = get_all_dimensions(topic, n_topic_dims)
            st.markdown("**Global (universal):**")
            for d in dims[:NUM_GLOBAL_DIMENSIONS]:
                st.markdown(f'<span class="dim-chip-global">{d["name"]}</span>', unsafe_allow_html=True)
            st.markdown("**Topic-specific:**")
            for d in dims[NUM_GLOBAL_DIMENSIONS:]:
                st.markdown(f'<span class="dim-chip-topic">{d["name"]}</span>', unsafe_allow_html=True)
            with st.expander("Details"):
                for d in dims:
                    tag = "🌐" if is_global_dimension(d["name"]) else "🔬"
                    st.markdown(f"{tag} **{d['name']}** — _{d['description']}_")
                    st.markdown(f"  🔻 {d['low']}  ·  🔺 {d['high']}")
        else:
            st.info("Click 🚀 Run Full Pipeline to generate dimensions.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">CARTOGRAPH</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Multi-layer research landscape navigator — map any field, discover gaps, surface directions.</p>', unsafe_allow_html=True)

if not topic:
    st.markdown("""
    <div class="onboard-card">
        <h3>👈 Enter a research topic to begin</h3>
        <div class="onboard-step">
            <div class="step-num">1</div>
            <div class="step-content"><strong>Scope</strong> — Enter a research field and choose how many latent dimensions to decompose papers into</div>
        </div>
        <div class="onboard-step">
            <div class="step-num">2</div>
            <div class="step-content"><strong>Ingest</strong> — System fetches papers from arXiv, scores each on all latent variables via LLM, and builds the multi-layer knowledge graph</div>
        </div>
        <div class="onboard-step">
            <div class="step-num">3</div>
            <div class="step-content"><strong>Query</strong> — Describe what you need in natural language, or directly tune the latent variable sliders to shape your ideal paper profile</div>
        </div>
        <div class="onboard-step">
            <div class="step-num">4</div>
            <div class="step-content"><strong>Discover</strong> — Get ranked papers with latent profile decompositions, radar charts, gap analysis, and AI-synthesized research directions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

topic_id = get_or_create_topic(topic, n_topic_dims)
all_papers = get_papers_by_topic(topic_id)
scored_papers = [p for p in all_papers if p["score_vector"]]

if not scored_papers:
    st.warning("⚠️ No scored papers yet. Run the pipeline from the sidebar.")
    st.stop()

import os as _os
if not _os.path.exists(_cache_path(topic)):
    st.warning("⚠️ Dimensions not generated yet. Run the pipeline first.")
    st.stop()

dims = get_all_dimensions(topic, n_topic_dims)
dim_names = [d["name"] for d in dims]

# Stats row
st.markdown("---")
cols = st.columns(5)
stats = [
    ("Papers", len(all_papers)), ("Scored", len(scored_papers)),
    ("Embedded", len([p for p in all_papers if p["embedding"]])),
    ("Global Dims", NUM_GLOBAL_DIMENSIONS), ("Topic Dims", len(dims) - NUM_GLOBAL_DIMENSIONS),
]
for col, (label, value) in zip(cols, stats):
    with col:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🔍 Query the Research Space")

query_mode = st.radio("Mode", ["Natural Language", "Latent Variable Control"], horizontal=True, label_visibility="collapsed")

desired_vector = None
priority_vector = None
user_query = ""

if query_mode == "Natural Language":
    user_query = st.text_area("Describe what you're looking for:",
        placeholder="e.g. I want something novel that doesn't need large datasets", height=100)
    if st.button("🔎 Search", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Translating query..."):
                try:
                    desired_vector, priority_vector = translate_query(topic, user_query, dims)
                    st.session_state["dv"] = desired_vector
                    st.session_state["pv"] = priority_vector
                    st.session_state["uq"] = user_query
                except Exception as e:
                    st.error(f"Translation failed: {e}")
    if "dv" in st.session_state:
        desired_vector = st.session_state["dv"]
        priority_vector = st.session_state["pv"]
        user_query = st.session_state["uq"]
else:
    st.markdown(
        '<div style="background:linear-gradient(135deg, rgba(139,92,246,0.08), rgba(99,202,255,0.04));'
        'border:1px solid rgba(139,92,246,0.2);'
        'border-radius:18px;padding:20px 24px;margin-bottom:20px;backdrop-filter:blur(10px);">'
        '<span style="font-size:1.15rem;font-weight:800;'
        'background:linear-gradient(135deg,#a78bfa,#63caff);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🧬 Latent Variable Control Panel</span><br/>'
        '<span style="color:#9ca3af;font-size:0.85rem;">'
        'Each paper is decomposed into these LLM-named latent variables. '
        'Tune the sliders to shape your ideal paper profile. <code style="color:#a78bfa;background:rgba(139,92,246,0.1);'
        'padding:2px 6px;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:0.8rem;">-1</code> = don\'t care.</span></div>',
        unsafe_allow_html=True)

    st.markdown("#### 🌐 Global Quality Axes")
    desired_vector = []
    gcols = st.columns(2)
    for i, dim in enumerate(dims[:NUM_GLOBAL_DIMENSIONS]):
        with gcols[i % 2]:
            st.markdown(
                f'<div style="background:linear-gradient(135deg, rgba(52,211,153,0.06), rgba(16,185,129,0.02));'
                f'border:1px solid rgba(52,211,153,0.15);'
                f'border-radius:14px;padding:14px 16px;margin-bottom:10px;transition:all 0.3s ease;">'
                f'<span style="color:#6ee7b7;font-weight:700;font-size:0.95rem;">🌐 {dim["name"]}</span>'
                f'<br/><span style="color:#9ca3af;font-size:0.8rem;line-height:1.4;">{dim["description"]}</span>'
                f'<div style="display:flex;justify-content:space-between;margin-top:6px;">'
                f'<span style="color:#374151;font-size:0.72rem;font-family:JetBrains Mono,monospace;">🔻 {dim["low"][:50]}</span>'
                f'<span style="color:#374151;font-size:0.72rem;font-family:JetBrains Mono,monospace;">🔺 {dim["high"][:50]}</span></div></div>',
                unsafe_allow_html=True)
            val = st.slider(dim["name"], -1.0, 1.0, -1.0, 0.1, key=f"g_{i}", label_visibility="collapsed")
            desired_vector.append(val)

    # Show ALL topic dimensions with pagination
    topic_all = dims[NUM_GLOBAL_DIMENSIONS:]
    page_size = 8
    n_pages = max(1, (len(topic_all) + page_size - 1) // page_size)
    st.markdown("#### 🔬 Topic-Specific Latent Variables")
    page = st.selectbox("Page", range(1, n_pages + 1), format_func=lambda p: f"Page {p}/{n_pages}", label_visibility="collapsed")
    page_start = (page - 1) * page_size
    page_dims = topic_all[page_start:page_start + page_size]
    tcols = st.columns(2)
    for i, dim in enumerate(page_dims):
        with tcols[i % 2]:
            st.markdown(
                f'<div style="background:linear-gradient(135deg, rgba(139,92,246,0.05), rgba(99,202,255,0.02));'
                f'border:1px solid rgba(139,92,246,0.12);'
                f'border-radius:14px;padding:14px 16px;margin-bottom:10px;transition:all 0.3s ease;">'
                f'<span style="color:#a78bfa;font-weight:700;font-size:0.95rem;">🔬 {dim["name"]}</span>'
                f'<br/><span style="color:#9ca3af;font-size:0.8rem;line-height:1.4;">{dim["description"]}</span>'
                f'<div style="display:flex;justify-content:space-between;margin-top:6px;">'
                f'<span style="color:#374151;font-size:0.72rem;font-family:JetBrains Mono,monospace;">🔻 {dim["low"][:50]}</span>'
                f'<span style="color:#374151;font-size:0.72rem;font-family:JetBrains Mono,monospace;">🔺 {dim["high"][:50]}</span></div></div>',
                unsafe_allow_html=True)
            key = f"sv_{NUM_GLOBAL_DIMENSIONS + page_start + i}"
            val = st.slider(dim["name"], -1.0, 1.0, -1.0, 0.1, key=key, label_visibility="collapsed")
            st.session_state[key] = val

    # Build full desired vector from session state
    for j in range(len(topic_all)):
        key = f"sv_{NUM_GLOBAL_DIMENSIONS + j}"
        desired_vector.append(st.session_state.get(key, -1.0))

    specified = [i for i, v in enumerate(desired_vector) if v != -1.0]
    n_spec = max(len(specified), 1)
    priority_vector = [(1.0/n_spec if i in specified else 0.0) for i in range(len(dims))]
    user_query = "Custom score vector query"

    if st.button("🔎 Search", type="primary", use_container_width=True):
        st.session_state["dv"] = desired_vector
        st.session_state["pv"] = priority_vector
        st.session_state["uq"] = user_query

# ── RESULTS ───────────────────────────────────────────────────────────────────
if desired_vector and priority_vector:
    st.markdown("---")

    with st.expander("📐 Translated Query Vectors"):
        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("**Desired**")
            for name, val in zip(dim_names, desired_vector):
                icon = "🌐" if is_global_dimension(name) else "🔬"
                st.markdown(f"{icon} `{name}`: **{val:.2f}**")
        with vc2:
            st.markdown("**Priority**")
            for name, val in zip(dim_names, priority_vector):
                bar = "█" * int(val * 20) if val > 0 else "·"
                st.markdown(f"`{name}`: **{val:.3f}** {bar}")

    with st.spinner("Searching multi-layer graph..."):
        results = retrieve_on_combined_graph(topic_id, desired_vector, priority_vector, dims)

    if not results:
        st.warning("No matching papers. Try building edges first.")
        st.stop()

    st.markdown(f'<div class="section-header"><div class="section-icon">📄</div><div class="section-text">Top {len(results)} Papers</div></div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="latent-legend latent-legend-global">🌐 Global axes</span>'
        '<span class="latent-legend latent-legend-topic">🔬 Topic variables</span>'
        '<span class="latent-legend latent-legend-desired">| Desired target</span>',
        unsafe_allow_html=True)

    # CSV Export button
    import csv, io
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(["rank", "title", "url", "distance"] + dim_names)
    for rank, p in enumerate(results, 1):
        writer.writerow([rank, p["title"], p["url"], f"{p['distance']:.4f}"] +
                        [f"{s:.3f}" for s in p["score_vector"]])
    st.download_button(
        "⬇️ Export results as CSV", csv_buf.getvalue(),
        file_name="cartograph_results.csv", mime="text/csv",
    )

    for paper in results:
        st.markdown(
            f'<div class="paper-card"><h4>{paper["title"]}</h4>'
            f'<span class="distance-badge">distance: {paper["distance"]:.4f}</span>'
            f'<br/><a class="paper-link" href="{paper["url"]}" target="_blank">📎 arXiv</a></div>',
            unsafe_allow_html=True)

        with st.expander(f"🧬 Latent Profile — {paper['title'][:50]}..."):
            st.markdown(
                '<div style="background:linear-gradient(135deg, rgba(139,92,246,0.06), rgba(99,202,255,0.03));'
                'border:1px solid rgba(139,92,246,0.12);'
                'border-radius:14px;padding:12px 16px;margin-bottom:14px;">'
                '<span style="color:#9ca3af;font-size:0.82rem;">'
                'This paper\'s identity is decomposed into the latent variables below. '
                'Each bar shows how strongly this paper expresses that variable.</span></div>',
                unsafe_allow_html=True)

            # Latent Profile: horizontal bar chart
            fig_lp, ax_lp = plt.subplots(figsize=(8, max(3, len(dim_names) * 0.35)))
            fig_lp.patch.set_facecolor("#06060b")
            ax_lp.set_facecolor("#06060b")
            y_pos = np.arange(len(dim_names))
            colors = ["#34d399" if j < NUM_GLOBAL_DIMENSIONS else "#a78bfa"
                       for j in range(len(dim_names))]
            ax_lp.barh(y_pos, paper["score_vector"], color=colors, height=0.6, alpha=0.85)
            # Overlay desired vector markers
            for j, dv_val in enumerate(desired_vector):
                if dv_val != -1.0:
                    ax_lp.plot(dv_val, j, marker='|', color='#fbbf24', markersize=18, markeredgewidth=2.5)
            ax_lp.set_yticks(y_pos)
            ax_lp.set_yticklabels(dim_names, fontsize=7, color="#d1d5db")
            ax_lp.set_xlim(0, 1)
            ax_lp.set_xlabel("Score", color="#9ca3af", fontsize=9)
            ax_lp.tick_params(axis='x', colors="#374151")
            ax_lp.invert_yaxis()
            ax_lp.set_title("Latent Variable Decomposition", color="#e5e7eb", fontsize=10, fontweight=600)
            for spine in ax_lp.spines.values():
                spine.set_color("#1f2937")
            ax_lp.grid(axis='x', color="#1f2937", linewidth=0.5, alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_lp)
            plt.close(fig_lp)

            # Radar chart
            fig = render_radar_chart(paper["score_vector"], desired_vector, dim_names, paper["title"][:40])
            st.pyplot(fig)
            plt.close(fig)

            # Full score table with dimension descriptions
            st.markdown("**All dimension scores:**")
            score_data = []
            for j, (dim_obj, sv) in enumerate(zip(dims, paper["score_vector"])):
                tier = "🌐 Global" if j < NUM_GLOBAL_DIMENSIONS else "🔬 Topic"
                dv = desired_vector[j] if desired_vector[j] != -1.0 else "—"
                gap = f"{abs(sv - desired_vector[j]):.2f}" if desired_vector[j] != -1.0 else "—"
                score_data.append({
                    "Tier": tier, "Dimension": dim_obj["name"],
                    "Description": dim_obj["description"][:60],
                    "Score": f"{sv:.3f}", "Desired": dv, "Gap": gap,
                })
            st.dataframe(score_data, use_container_width=True, hide_index=True)

    # Gap Analysis
    st.markdown('---')
    st.markdown('<div class="section-header"><div class="section-icon">🕳️</div><div class="section-text">Gap Analysis</div></div>', unsafe_allow_html=True)
    gap_info = detect_gap(load_graph(topic_id), results, dims, desired_vector)

    if gap_info["global_gaps"]:
        st.markdown("**🌐 Global quality gaps:**")
        for g in gap_info["global_gaps"]:
            st.markdown(f'<span class="gap-pill-global">{g["dimension"]} — desired: {g["desired"]:.2f}, available: {g["available"]:.2f} (Δ{g["gap"]:.2f})</span>', unsafe_allow_html=True)
    if gap_info["topic_gaps"]:
        st.markdown("**🔬 Topic-specific gaps:**")
        for g in gap_info["topic_gaps"]:
            st.markdown(f'<span class="gap-pill">{g["dimension"]} — desired: {g["desired"]:.2f}, available: {g["available"]:.2f} (Δ{g["gap"]:.2f})</span>', unsafe_allow_html=True)
    if not gap_info["global_gaps"] and not gap_info["topic_gaps"]:
        st.success("No significant gaps — literature covers your query well!")

    # Dimension Correlation Heatmap
    st.markdown('---')
    st.markdown('<div class="section-header"><div class="section-icon">🔥</div><div class="section-text">Dimension Correlations</div></div>', unsafe_allow_html=True)
    if len(results) >= 3:
        score_mat = np.array([p["score_vector"] for p in scored_papers if len(json.loads(p["score_vector"]) if isinstance(p["score_vector"], str) else p["score_vector"]) == len(dim_names)])
        if len(score_mat) >= 3:
            corr = np.corrcoef(score_mat.T)
            fig_h, ax_h = plt.subplots(figsize=(10, 8))
            fig_h.patch.set_facecolor("#06060b")
            ax_h.set_facecolor("#06060b")
            # Custom colormap for the dark theme
            from matplotlib.colors import LinearSegmentedColormap
            cmap_custom = LinearSegmentedColormap.from_list("cartograph",
                ["#6d28d9", "#1f2937", "#06060b", "#1f2937", "#34d399"])
            im = ax_h.imshow(corr, cmap=cmap_custom, vmin=-1, vmax=1, aspect="auto")
            ax_h.set_xticks(range(len(dim_names)))
            ax_h.set_yticks(range(len(dim_names)))
            ax_h.set_xticklabels(dim_names, rotation=45, ha="right", fontsize=6, color="#9ca3af")
            ax_h.set_yticklabels(dim_names, fontsize=6, color="#9ca3af")
            cbar = fig_h.colorbar(im, ax=ax_h, shrink=0.8)
            cbar.set_label("Pearson r", color="#9ca3af")
            cbar.ax.yaxis.set_tick_params(color="#374151")
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#9ca3af")
            ax_h.set_title("Dimension Correlations Across All Scored Papers", color="#e5e7eb", fontsize=11, fontweight=600)
            st.pyplot(fig_h)
            plt.close(fig_h)
    else:
        st.info("Need at least 3 results for correlation heatmap.")

    # Synthesis
    st.markdown('---')
    st.markdown('<div class="section-header"><div class="section-icon">💡</div><div class="section-text">Synthesized Research Directions</div></div>', unsafe_allow_html=True)
    with st.spinner("Synthesizing directions from gaps..."):
        directions = synthesize_directions(topic, user_query, results, gap_info, dims)
    if directions:
        for d in directions:
            st.markdown(
                f'<div class="direction-card"><h4>🧭 {d.get("title","Untitled")}</h4>'
                f'<p>{d.get("description","")}</p>'
                f'<p><strong>Gap rationale:</strong> {d.get("gap_rationale","")}</p>'
                f'<div class="experiment-box">🧪 <strong>First experiment:</strong> {d.get("first_experiment","N/A")}</div></div>',
                unsafe_allow_html=True)
    else:
        st.info("No directions synthesized — try refining your query.")

    # Graph Visualization + Stats
    st.markdown('---')
    st.markdown('<div class="section-header"><div class="section-icon">🕸️</div><div class="section-text">Knowledge Graph</div></div>', unsafe_allow_html=True)

    G_combined = load_combined_graph(topic_id, priority_vector, dims)
    G_semantic = load_graph(topic_id)

    gc1, gc2, gc3, gc4 = st.columns(4)
    with gc1: st.metric("Nodes", G_combined.number_of_nodes())
    with gc2: st.metric("Combined Edges", G_combined.number_of_edges())
    with gc3: st.metric("Semantic Edges", G_semantic.number_of_edges())
    with gc4: st.metric("Density", f"{nx.density(G_combined):.4f}" if G_combined.number_of_nodes() > 0 else "N/A")

    if G_combined.number_of_nodes() > 0:
        comps = list(nx.connected_components(G_combined))
        st.markdown(f"**Components:** {len(comps)} · **Largest:** {len(max(comps, key=len))} nodes")

    if G_combined.number_of_nodes() > 0:
        try:
            from pyvis.network import Network
            import streamlit.components.v1 as components
            import tempfile

            result_ids = {p["id"] for p in results}

            net = Network(
                height="600px", width="100%",
                bgcolor="#06060b", font_color="#d1d5db",
                directed=False,
            )
            net.barnes_hut(
                gravity=-3000,
                central_gravity=0.3,
                spring_length=120,
                spring_strength=0.01,
                damping=0.09,
            )

            for node_id, data in G_combined.nodes(data=True):
                title_text = data.get("title", "Unknown")[:60]
                sv = data.get("score_vector", [])
                score_preview = ", ".join(f"{s:.1f}" for s in sv[:6])
                if len(sv) > 6:
                    score_preview += "..."
                hover = f"<b>{title_text}</b><br>Scores: [{score_preview}]"

                if node_id in result_ids:
                    net.add_node(
                        node_id, label=title_text[:25],
                        title=hover,
                        color="#a78bfa", size=22,
                        borderWidth=2, borderWidthSelected=4,
                    )
                else:
                    net.add_node(
                        node_id, label="",
                        title=hover,
                        color="#1f2937", size=8,
                        borderWidth=1,
                    )

            for u, v, edata in G_combined.edges(data=True):
                w = edata.get("weight", 0.1)
                sem_w = edata.get("semantic_weight", 0)
                dim_w = edata.get("dimension_weight", 0)

                if sem_w > 0 and dim_w > 0:
                    color = "#34d399"  # both layers — emerald
                elif sem_w > 0:
                    color = "#63caff"  # semantic only — cyan
                else:
                    color = "#374151"  # dimension only — muted

                net.add_edge(
                    u, v,
                    value=max(w * 3, 0.5),
                    color=color,
                    title=f"W={w:.3f} (sem={sem_w:.2f}, dim={dim_w:.2f})",
                )

            import os
            graph_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data", "graph_vis.html"
            )
            net.save_graph(graph_path)

            with open(graph_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            st.markdown(
                '<span class="latent-legend latent-legend-topic">🟣 Retrieved</span>'
                '<span class="latent-legend" style="background:rgba(31,41,55,0.5);color:#6b7280;border:1px solid #374151;">⚪ Other</span>'
                '<span class="latent-legend latent-legend-global">🟢 Both layers</span>'
                '<span class="latent-legend" style="background:rgba(99,202,255,0.1);color:#63caff;border:1px solid rgba(99,202,255,0.25);">🔵 Semantic</span>'
                '<span class="latent-legend" style="background:rgba(55,65,81,0.3);color:#6b7280;border:1px solid #374151;">⚫ Dimension</span>',
                unsafe_allow_html=True)
            components.html(html_content, height=620, scrolling=False)

        except ImportError:
            st.warning("Install pyvis for graph visualization: `pip install pyvis`")
        except Exception as e:
            st.error(f"Graph visualization failed: {e}")
