# db.py — SQLite schema + CRUD helpers for CARTOGRAPH v2
import sqlite3
import json
from config import DB_PATH


def _get_conn() -> sqlite3.Connection:
    """Return a SQLite connection with row-factory enabled."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
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
            abstract TEXT NOT NULL,
            authors TEXT,
            published TEXT,
            url TEXT,
            score_vector TEXT,
            embedding TEXT,
            FOREIGN KEY (topic_id) REFERENCES topics(id)
        );

        CREATE TABLE IF NOT EXISTS edges (
            paper_a TEXT NOT NULL,
            paper_b TEXT NOT NULL,
            weight REAL NOT NULL,
            edge_type TEXT NOT NULL,
            PRIMARY KEY (paper_a, paper_b)
        );

        CREATE TABLE IF NOT EXISTS dimension_edges (
            paper_a TEXT NOT NULL,
            paper_b TEXT NOT NULL,
            dimension TEXT NOT NULL,
            weight REAL NOT NULL,
            PRIMARY KEY (paper_a, paper_b, dimension)
        );

        CREATE INDEX IF NOT EXISTS idx_dim_edges_dim
            ON dimension_edges(dimension);
    """)
    conn.commit()
    conn.close()


def get_or_create_topic(name: str, num_topic_dims: int = 20) -> int:
    """Return the topic id, creating a new row if it doesn't exist."""
    conn = _get_conn()
    cur = conn.execute("SELECT id FROM topics WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        topic_id = row["id"]
    else:
        cur = conn.execute(
            "INSERT INTO topics (name, num_topic_dims) VALUES (?, ?)",
            (name, num_topic_dims),
        )
        conn.commit()
        topic_id = cur.lastrowid
    conn.close()
    return topic_id


def get_topic_dim_count(topic_id: int) -> int:
    """Return the number of topic-specific dimensions for a topic."""
    conn = _get_conn()
    cur = conn.execute("SELECT num_topic_dims FROM topics WHERE id = ?", (topic_id,))
    row = cur.fetchone()
    conn.close()
    return row["num_topic_dims"] if row else 20


def insert_paper(paper: dict) -> None:
    """Upsert a paper record (insert or ignore on existing id)."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR IGNORE INTO papers
           (id, topic_id, title, abstract, authors, published, url)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            paper["id"],
            paper["topic_id"],
            paper["title"],
            paper["abstract"],
            paper.get("authors"),
            paper.get("published"),
            paper.get("url"),
        ),
    )
    conn.commit()
    conn.close()


def get_papers_by_topic(topic_id: int) -> list[dict]:
    """Return all papers for a topic as a list of dicts."""
    conn = _get_conn()
    cur = conn.execute("SELECT * FROM papers WHERE topic_id = ?", (topic_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def update_score_vector(paper_id: str, vector: list[float]) -> None:
    """Store the score vector as a JSON string."""
    conn = _get_conn()
    conn.execute(
        "UPDATE papers SET score_vector = ? WHERE id = ?",
        (json.dumps(vector), paper_id),
    )
    conn.commit()
    conn.close()


def update_embedding(paper_id: str, embedding: list[float]) -> None:
    """Store the SBERT embedding as a JSON string."""
    conn = _get_conn()
    conn.execute(
        "UPDATE papers SET embedding = ? WHERE id = ?",
        (json.dumps(embedding), paper_id),
    )
    conn.commit()
    conn.close()


def insert_edge(a: str, b: str, weight: float, edge_type: str) -> None:
    """Insert or replace a semantic graph edge."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO edges (paper_a, paper_b, weight, edge_type) VALUES (?, ?, ?, ?)",
        (a, b, weight, edge_type),
    )
    conn.commit()
    conn.close()


def insert_dimension_edge(a: str, b: str, dimension: str, weight: float) -> None:
    """Insert or replace a per-dimension graph edge."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO dimension_edges (paper_a, paper_b, dimension, weight) VALUES (?, ?, ?, ?)",
        (a, b, dimension, weight),
    )
    conn.commit()
    conn.close()


def insert_dimension_edges_bulk(edges: list[tuple]) -> None:
    """Bulk insert per-dimension edges. Each tuple: (paper_a, paper_b, dimension, weight)."""
    conn = _get_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO dimension_edges (paper_a, paper_b, dimension, weight) VALUES (?, ?, ?, ?)",
        edges,
    )
    conn.commit()
    conn.close()


def get_edges_by_topic(topic_id: int) -> list[dict]:
    """Return all semantic edges where at least one endpoint belongs to the topic."""
    conn = _get_conn()
    cur = conn.execute(
        """SELECT e.* FROM edges e
           JOIN papers p ON e.paper_a = p.id
           WHERE p.topic_id = ?""",
        (topic_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_dimension_edges_by_topic(topic_id: int) -> list[dict]:
    """Return all per-dimension edges where at least one endpoint belongs to the topic."""
    conn = _get_conn()
    cur = conn.execute(
        """SELECT de.* FROM dimension_edges de
           JOIN papers p ON de.paper_a = p.id
           WHERE p.topic_id = ?""",
        (topic_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_dimension_edges_for_dim(topic_id: int, dimension: str) -> list[dict]:
    """Return edges for a specific dimension within a topic."""
    conn = _get_conn()
    cur = conn.execute(
        """SELECT de.* FROM dimension_edges de
           JOIN papers p ON de.paper_a = p.id
           WHERE p.topic_id = ? AND de.dimension = ?""",
        (topic_id, dimension),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_unscored_papers(topic_id: int) -> list[dict]:
    """Return papers that have not yet been scored."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT * FROM papers WHERE topic_id = ? AND score_vector IS NULL",
        (topic_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_unembedded_papers(topic_id: int) -> list[dict]:
    """Return papers that have not yet been embedded."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT * FROM papers WHERE topic_id = ? AND embedding IS NULL",
        (topic_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# Initialize tables on import
init_db()
