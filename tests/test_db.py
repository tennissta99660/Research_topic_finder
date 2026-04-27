# tests/test_db.py — Database round-trip tests
import json
import sqlite3
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDatabase:
    """Test SQLite operations via direct SQL (avoids import-time side effects of db.py)."""

    def test_create_topic(self, temp_db):
        conn = sqlite3.connect(temp_db)
        conn.execute("INSERT INTO topics (name, num_topic_dims) VALUES (?, ?)", ("ml", 20))
        conn.commit()
        row = conn.execute("SELECT name, num_topic_dims FROM topics WHERE name='ml'").fetchone()
        assert row == ("ml", 20)
        conn.close()

    def test_duplicate_topic_rejected(self, temp_db):
        conn = sqlite3.connect(temp_db)
        conn.execute("INSERT INTO topics (name) VALUES (?)", ("ml",))
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO topics (name) VALUES (?)", ("ml",))
        conn.close()

    def test_insert_and_retrieve_paper(self, temp_db, sample_score_vector):
        conn = sqlite3.connect(temp_db)
        conn.execute("INSERT INTO topics (id, name) VALUES (1, 'ml')")
        sv = json.dumps(sample_score_vector)
        conn.execute(
            "INSERT INTO papers (id, topic_id, title, abstract, url, score_vector) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("p1", 1, "Test Paper", "Abstract here", "http://example.com", sv),
        )
        conn.commit()
        row = conn.execute("SELECT id, title, score_vector FROM papers WHERE id='p1'").fetchone()
        assert row[0] == "p1"
        assert row[1] == "Test Paper"
        assert json.loads(row[2]) == sample_score_vector
        conn.close()

    def test_score_vector_roundtrip_preserves_precision(self, temp_db):
        conn = sqlite3.connect(temp_db)
        conn.execute("INSERT INTO topics (id, name) VALUES (1, 'ml')")
        vec = [0.123456789, 0.987654321, 0.555555555]
        conn.execute(
            "INSERT INTO papers (id, topic_id, title, score_vector) VALUES (?, ?, ?, ?)",
            ("p1", 1, "Test", json.dumps(vec)),
        )
        conn.commit()
        row = conn.execute("SELECT score_vector FROM papers WHERE id='p1'").fetchone()
        loaded = json.loads(row[0])
        for a, b in zip(vec, loaded):
            assert abs(a - b) < 1e-9
        conn.close()

    def test_dimension_edge_bulk_insert(self, temp_db):
        conn = sqlite3.connect(temp_db)
        conn.execute("INSERT INTO topics (id, name) VALUES (1, 'ml')")
        conn.execute("INSERT INTO papers (id, topic_id, title) VALUES ('a', 1, 'A')")
        conn.execute("INSERT INTO papers (id, topic_id, title) VALUES ('b', 1, 'B')")
        conn.commit()

        edges = [("a", "b", "scalability", 0.85), ("a", "b", "efficiency", 0.72)]
        conn.executemany(
            "INSERT OR REPLACE INTO dimension_edges (paper_a, paper_b, dimension, weight) "
            "VALUES (?, ?, ?, ?)",
            edges,
        )
        conn.commit()

        rows = conn.execute("SELECT * FROM dimension_edges").fetchall()
        assert len(rows) == 2
        dims = {r[2] for r in rows}
        assert dims == {"scalability", "efficiency"}
        conn.close()

    def test_edge_composite_pk(self, temp_db):
        conn = sqlite3.connect(temp_db)
        conn.execute("INSERT INTO topics (id, name) VALUES (1, 'ml')")
        conn.execute("INSERT INTO papers (id, topic_id, title) VALUES ('a', 1, 'A')")
        conn.execute("INSERT INTO papers (id, topic_id, title) VALUES ('b', 1, 'B')")
        conn.execute(
            "INSERT INTO edges (paper_a, paper_b, weight, edge_type) VALUES ('a', 'b', 0.9, 'semantic')"
        )
        conn.commit()
        # Same pair + type should conflict
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO edges (paper_a, paper_b, weight, edge_type) VALUES ('a', 'b', 0.8, 'semantic')"
            )
        conn.close()
