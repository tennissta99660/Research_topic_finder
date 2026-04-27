# tests/test_query.py — Query module tests
import os, sys, pytest
import networkx as nx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from query import weighted_distance, retrieve

class TestWeightedDistance:
    def test_perfect_match(self):
        d = weighted_distance([0.8,0.5], [0.8,0.5], [0.5,0.5])
        assert abs(d) < 1e-6

    def test_unspecified_dims(self):
        d = weighted_distance([0.8], [-1.0], [0.0])
        assert d == pytest.approx(0.1 * abs(0.8 - 0.5), abs=1e-6)

    def test_priority_weighting(self):
        d = weighted_distance([0.0, 1.0], [1.0, 1.0], [0.9, 0.1])
        assert d == pytest.approx(0.9 * 1.0 + 0.1 * 0.0, abs=1e-6)

class TestRetrieve:
    def test_returns_ascending(self):
        G = nx.Graph()
        G.add_node("a", score_vector=[0.9, 0.9])
        G.add_node("b", score_vector=[0.5, 0.5])
        G.add_node("c", score_vector=[0.1, 0.1])
        results = retrieve(G, [0.9, 0.9], [0.5, 0.5], top_k=3)
        assert results[0][0] == "a"  # closest
        assert results[-1][0] == "c"  # farthest

    def test_top_k_limits(self):
        G = nx.Graph()
        for i in range(10):
            G.add_node(f"p{i}", score_vector=[i*0.1])
        results = retrieve(G, [0.5], [1.0], top_k=3)
        assert len(results) == 3
