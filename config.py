# config.py — All constants and thresholds for CARTOGRAPH v2
import os
import warnings
import logging

# Suppress transformers __path__ deprecation noise (must be before any HF imports)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="Accessing.*__path__")
logging.getLogger("transformers").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

# ── API Keys (up to 8, auto-rotates on rate limit) ────────────────────────────
GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY_1", os.getenv("GROQ_API_KEY", "")),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
    os.getenv("GROQ_API_KEY_4", ""),
    os.getenv("GROQ_API_KEY_5", ""),
    os.getenv("GROQ_API_KEY_6", ""),
    os.getenv("GROQ_API_KEY_7", ""),
    os.getenv("GROQ_API_KEY_8", ""),
]
# Filter out empty keys
GROQ_API_KEYS = [k for k in GROQ_API_KEYS if k.strip()]

# ── Paper Ingestion ───────────────────────────────────────────────────────────
ARXIV_MAX_RESULTS = 200          # papers per topic fetch

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EDGE_SIMILARITY_THRESHOLD = 0.50  # for semantic edges (was 0.75, too strict)

# ── LLM Backend ──────────────────────────────────────────────────────────────
# Priority: Lightning AI (remote) → Local Ollama → Groq cloud
#
# Lightning AI: remote Ollama on cloud GPU (primary)
# Local Ollama: fallback when Lightning AI is unavailable
# Groq:        last-resort cloud fallback (rate limited)

# ── Lightning AI (primary — remote Ollama on cloud GPU) ──────────────────────
LIGHTNING_OLLAMA_URL = os.getenv("LIGHTNING_OLLAMA_URL", "")
LIGHTNING_MODEL = os.getenv("LIGHTNING_MODEL", "qwen3:32b")

# ── Local Ollama (fallback) ──────────────────────────────────────────────────
LOCAL_OLLAMA_URL = os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434/v1")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen3:8b")

# ── Groq (last-resort cloud fallback) ───────────────────────────────────────
SCORE_MODEL = os.getenv("SCORE_MODEL", "llama-3.1-8b-instant")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "llama-3.1-8b-instant")

# ── Dimensions — Two-Tier System ──────────────────────────────────────────────
NUM_GLOBAL_DIMENSIONS = 4         # fixed: novelty, rigor, impact, reproducibility
MAX_DIMENSIONS = 50               # total (global + topic-specific)
MIN_TOPIC_DIMENSIONS = 6          # minimum topic-specific dimensions
MAX_TOPIC_DIMENSIONS = 46         # MAX_DIMENSIONS - NUM_GLOBAL_DIMENSIONS
DEFAULT_TOPIC_DIMENSIONS = 10     # default topic-specific count (lower for local LLMs)

# ── Global Dimensions (universal across all topics) ───────────────────────────
GLOBAL_DIMENSIONS = [
    {
        "name": "novelty",
        "description": "How novel or original the research contribution is",
        "low": "Incremental extension of existing work with no new ideas",
        "high": "Introduces a fundamentally new concept, method, or paradigm",
    },
    {
        "name": "rigor",
        "description": "Methodological soundness and thoroughness of evaluation",
        "low": "No formal evaluation, weak baselines, or flawed methodology",
        "high": "Comprehensive evaluation with strong baselines, ablations, and statistical analysis",
    },
    {
        "name": "impact",
        "description": "Potential significance and influence on the field",
        "low": "Narrow application with limited broader relevance",
        "high": "Foundational contribution likely to influence many subsequent works",
    },
    {
        "name": "reproducibility",
        "description": "How easily the results can be reproduced by others",
        "low": "No code, unclear methodology, proprietary data",
        "high": "Open-source code, public datasets, detailed implementation notes",
    },
]

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 20             # candidates before graph traversal
TOP_K_OUTPUT = 5                 # final output papers
GRAPH_HOP_DEPTH = 5              # neighborhood traversal depth

# ── Query Weighting ───────────────────────────────────────────────────────────
SECONDARY_DIM_WEIGHT = 0.1       # ε for flexible (unspecified) dimensions

# ── Graph — Multi-Layer Merging ───────────────────────────────────────────────
SEMANTIC_EDGE_ALPHA = 0.3        # weight of semantic edges in combined graph
DIM_EDGE_THRESHOLD = 0.7         # min score proximity (1-|diff|) for per-dimension edges
GLOBAL_DIM_FILTER_THRESHOLD = 0.0  # papers below this on any global dim are excluded (0=no filter)

# ── Gap Detection ─────────────────────────────────────────────────────────────
GAP_MIN_NEIGHBOR_DIST = 0.3      # min distance to call a region sparse

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DIMENSIONS_DIR = os.path.join(DATA_DIR, "dimensions")
DB_PATH = os.path.join(DATA_DIR, "cartograph.db")

# Ensure data directories exist
os.makedirs(DIMENSIONS_DIR, exist_ok=True)
