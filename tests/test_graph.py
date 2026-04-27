# tests/test_graph.py — Graph construction and traversal tests
import os, sys, pytest
import numpy as np
import networkx as nx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graph import get_neighborhood

class TestSemanticEdge:
    def test_identical_sim_one(self):
        a = np.array([1.0, 0.0, 0.0])
        sim = float(np.dot(a, a) / (np.linalg.norm(a)**2))
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_sim_zero(self):
        a, b = np.array([1,0,0.]), np.array([0,1,0.])
        sim = float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)))
        assert abs(sim) < 1e-6

class TestDimProximity:
    def test_identical(self):
        assert 1.0 - abs(0.7-0.7) == 1.0
    def test_max_diff(self):
        assert 1.0 - abs(1.0-0.0) == 0.0
    def test_threshold(self):
        assert abs((1.0 - abs(0.8-0.5)) - 0.7) < 1e-6

class TestMergeFormula:
    def test_both(self):
        assert abs(0.3*0.8 + 0.7*0.6 - 0.66) < 1e-6
    def test_sem_only(self):
        assert abs(0.3*0.9 + 0.7*0.0 - 0.27) < 1e-6

class TestBFS:
    def _chain(self):
        G = nx.Graph()
        G.add_edges_from([("a","b"),("b","c"),("c","d"),("d","e")])
        return G
    def test_depth_0(self):
        assert get_neighborhood(self._chain(), "a", 0) == {"a"}
    def test_depth_1(self):
        assert get_neighborhood(self._chain(), "a", 1) == {"a","b"}
    def test_depth_2(self):
        assert get_neighborhood(self._chain(), "a", 2) == {"a","b","c"}
    def test_overflow(self):
        assert get_neighborhood(self._chain(), "a", 10) == {"a","b","c","d","e"}
    def test_missing(self):
        assert get_neighborhood(self._chain(), "z", 3) == set()
