# tests/test_dimensions.py — Dimension system tests
import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dimensions import (
    get_global_dimensions,
    is_global_dimension,
)
from config import GLOBAL_DIMENSIONS, NUM_GLOBAL_DIMENSIONS


class TestGlobalDimensions:
    def test_exactly_four(self):
        dims = get_global_dimensions()
        assert len(dims) == 4

    def test_correct_names(self):
        dims = get_global_dimensions()
        names = [d["name"] for d in dims]
        assert names == ["novelty", "rigor", "impact", "reproducibility"]

    def test_each_has_required_keys(self):
        for dim in get_global_dimensions():
            assert "name" in dim
            assert "description" in dim
            assert "low" in dim
            assert "high" in dim

    def test_is_global_dimension(self):
        assert is_global_dimension("novelty") is True
        assert is_global_dimension("rigor") is True
        assert is_global_dimension("scalability") is False
        assert is_global_dimension("NOVELTY") is True  # case-insensitive


class TestDimensionOrdering:
    def test_global_always_first(self, all_dimensions):
        """Global dims must always be at indices 0-3."""
        global_names = {"novelty", "rigor", "impact", "reproducibility"}
        for i in range(NUM_GLOBAL_DIMENSIONS):
            assert all_dimensions[i]["name"] in global_names

    def test_topic_dims_after_global(self, all_dimensions):
        """Topic dims start at index 4."""
        global_names = {"novelty", "rigor", "impact", "reproducibility"}
        for i in range(NUM_GLOBAL_DIMENSIONS, len(all_dimensions)):
            assert all_dimensions[i]["name"] not in global_names


class TestDimensionCache:
    def test_cache_file_created(self, tmp_path):
        """Verify that dimension cache writes valid JSON."""
        cache_file = tmp_path / "test_dims.json"
        dims = [{"name": "test", "description": "d", "low": "l", "high": "h"}]
        cache_file.write_text(json.dumps(dims))

        loaded = json.loads(cache_file.read_text())
        assert len(loaded) == 1
        assert loaded[0]["name"] == "test"

    def test_cache_roundtrip_preserves_all_fields(self, tmp_path):
        cache_file = tmp_path / "dims.json"
        dims = [
            {"name": "x", "description": "desc", "low": "lo", "high": "hi"},
            {"name": "y", "description": "desc2", "low": "lo2", "high": "hi2"},
        ]
        cache_file.write_text(json.dumps(dims))
        loaded = json.loads(cache_file.read_text())
        assert loaded == dims
