# tests/conftest.py — Shared fixtures for CARTOGRAPH test suite
import os
import sys
import json
import pytest
import sqlite3
import tempfile

# Add parent directory to path so we can import cartograph modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_global_dimensions():
    """The 4 hardcoded global dimensions."""
    return [
        {"name": "novelty", "description": "How novel", "low": "Not novel", "high": "Very novel"},
        {"name": "rigor", "description": "How rigorous", "low": "Weak", "high": "Strong"},
        {"name": "impact", "description": "How impactful", "low": "Low", "high": "High"},
        {"name": "reproducibility", "description": "How reproducible", "low": "Hard", "high": "Easy"},
    ]


@pytest.fixture
def sample_topic_dimensions():
    """6 sample topic-specific dimensions."""
    return [
        {"name": "scalability", "description": "Scales to large inputs", "low": "Small only", "high": "Web-scale"},
        {"name": "interpretability", "description": "Results are explainable", "low": "Black box", "high": "Fully transparent"},
        {"name": "efficiency", "description": "Computational cost", "low": "Very expensive", "high": "Real-time"},
        {"name": "generalization", "description": "Cross-domain transfer", "low": "Task-specific", "high": "Universal"},
        {"name": "robustness", "description": "Handles noise", "low": "Fragile", "high": "Robust"},
        {"name": "data_efficiency", "description": "Sample complexity", "low": "Needs millions", "high": "Few-shot"},
    ]


@pytest.fixture
def all_dimensions(sample_global_dimensions, sample_topic_dimensions):
    """Combined global + topic dimensions (10 total)."""
    return sample_global_dimensions + sample_topic_dimensions


@pytest.fixture
def sample_score_vector():
    """A 10-dimension score vector (4 global + 6 topic)."""
    return [0.8, 0.7, 0.6, 0.5, 0.9, 0.3, 0.7, 0.4, 0.8, 0.6]


@pytest.fixture
def sample_papers(sample_score_vector):
    """List of mock paper dicts."""
    return [
        {
            "id": "paper_001",
            "title": "Attention Is All You Need",
            "abstract": "We propose a new architecture based on attention mechanisms.",
            "url": "https://arxiv.org/abs/1706.03762",
            "score_vector": json.dumps(sample_score_vector),
            "embedding": json.dumps([0.1] * 384),
        },
        {
            "id": "paper_002",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce BERT for language understanding.",
            "url": "https://arxiv.org/abs/1810.04805",
            "score_vector": json.dumps([0.7, 0.8, 0.9, 0.6, 0.5, 0.4, 0.8, 0.3, 0.7, 0.5]),
            "embedding": json.dumps([0.2] * 384),
        },
        {
            "id": "paper_003",
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "abstract": "We show that scaling up language models improves few-shot performance.",
            "url": "https://arxiv.org/abs/2005.14165",
            "score_vector": json.dumps([0.9, 0.6, 0.8, 0.3, 0.8, 0.2, 0.6, 0.9, 0.5, 0.7]),
            "embedding": json.dumps([0.15] * 384),
        },
    ]


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database with the cartograph schema."""
    db_path = str(tmp_path / "test_cartograph.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            num_topic_dims INTEGER DEFAULT 20,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            topic_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            abstract TEXT,
            url TEXT,
            score_vector TEXT,
            embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (topic_id) REFERENCES topics(id)
        );
        CREATE TABLE IF NOT EXISTS edges (
            paper_a TEXT NOT NULL,
            paper_b TEXT NOT NULL,
            weight REAL NOT NULL,
            edge_type TEXT DEFAULT 'semantic',
            PRIMARY KEY (paper_a, paper_b, edge_type),
            FOREIGN KEY (paper_a) REFERENCES papers(id),
            FOREIGN KEY (paper_b) REFERENCES papers(id)
        );
        CREATE TABLE IF NOT EXISTS dimension_edges (
            paper_a TEXT NOT NULL,
            paper_b TEXT NOT NULL,
            dimension TEXT NOT NULL,
            weight REAL NOT NULL,
            PRIMARY KEY (paper_a, paper_b, dimension),
            FOREIGN KEY (paper_a) REFERENCES papers(id),
            FOREIGN KEY (paper_b) REFERENCES papers(id)
        );
        CREATE INDEX IF NOT EXISTS idx_dim_edges_dim ON dimension_edges(dimension);
    """)
    conn.close()
    return db_path
