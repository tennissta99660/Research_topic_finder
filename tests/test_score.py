# tests/test_score.py — Scoring module tests
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from score import _parse_score_response


class TestParseScoreResponse:
    def test_valid_response(self, all_dimensions):
        raw = json.dumps({d["name"]: 0.5 for d in all_dimensions})
        result = _parse_score_response(raw, all_dimensions)
        assert result is not None
        assert len(result) == len(all_dimensions)
        assert all(v == 0.5 for v in result)

    def test_preserves_ordering(self, all_dimensions):
        """Scores must be ordered to match dimension list, not alphabetical."""
        scores = {d["name"]: round(i * 0.1, 1) for i, d in enumerate(all_dimensions)}
        raw = json.dumps(scores)
        result = _parse_score_response(raw, all_dimensions)
        assert result is not None
        for i, dim in enumerate(all_dimensions):
            assert result[i] == scores[dim["name"]]

    def test_missing_dimension_returns_none(self, all_dimensions):
        scores = {d["name"]: 0.5 for d in all_dimensions[:-1]}  # missing last
        raw = json.dumps(scores)
        result = _parse_score_response(raw, all_dimensions)
        assert result is None

    def test_clamps_above_one(self, all_dimensions):
        scores = {d["name"]: 1.5 for d in all_dimensions}
        raw = json.dumps(scores)
        result = _parse_score_response(raw, all_dimensions)
        assert result is not None
        assert all(v == 1.0 for v in result)

    def test_clamps_below_zero(self, all_dimensions):
        scores = {d["name"]: -0.3 for d in all_dimensions}
        raw = json.dumps(scores)
        result = _parse_score_response(raw, all_dimensions)
        assert result is not None
        assert all(v == 0.0 for v in result)

    def test_malformed_json_returns_none(self, all_dimensions):
        result = _parse_score_response("not json at all", all_dimensions)
        assert result is None

    def test_non_dict_json_returns_none(self, all_dimensions):
        result = _parse_score_response("[1, 2, 3]", all_dimensions)
        assert result is None

    def test_non_numeric_value_returns_none(self, all_dimensions):
        scores = {d["name"]: "high" for d in all_dimensions}
        raw = json.dumps(scores)
        result = _parse_score_response(raw, all_dimensions)
        assert result is None
