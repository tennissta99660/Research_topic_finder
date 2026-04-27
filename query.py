# query.py — Query translation + weighted retrieval for CARTOGRAPH v2
#
# Uses the combined multi-layer graph for retrieval.
# Global dimensions can act as hard filters.
import json
import logging

import networkx as nx

from config import (
    SCORE_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_OUTPUT,
    SECONDARY_DIM_WEIGHT,
    NUM_GLOBAL_DIMENSIONS,
)
from llm import call_llm
from utils import format_dimensions_text as _format_dimensions_text
from graph import get_neighborhood, load_combined_graph

logger = logging.getLogger(__name__)


QUERY_TRANSLATION_PROMPT = """
The user wants to find research papers or ideas in the field of {topic}.

The field is characterized by TWO tiers of dimensions:

=== GLOBAL DIMENSIONS (universal research quality axes) ===
{global_dimensions_text}

=== TOPIC-SPECIFIC DIMENSIONS (field-specific technical axes) ===
{topic_dimensions_text}

User's request: "{user_query}"

Produce two vectors, each of length {n_dims} (covering ALL dimensions in order: global first, then topic-specific):

1. desired_vector: For each dimension, the target score the user wants (0.0–1.0).
   Use -1.0 for dimensions the user has not expressed a preference about.

2. priority_vector: For each dimension the user DID specify, assign a weight.
   Weights must sum to 1.0. Unspecified dimensions (desired=-1) get weight 0.0.

Return ONLY JSON:
{{
  "desired_vector": [...],
  "priority_vector": [...]
}}
"""




def translate_query(
    topic: str,
    user_query: str,
    dimensions: list[dict],
) -> tuple[list[float], list[float]]:
    """Translate a natural-language query into (desired_vector, priority_vector).

    Returns fallback uniform vectors if LLM response is malformed.
    """
    n = len(dimensions)
    global_dims = dimensions[:NUM_GLOBAL_DIMENSIONS]
    topic_dims = dimensions[NUM_GLOBAL_DIMENSIONS:]

    prompt = QUERY_TRANSLATION_PROMPT.format(
        topic=topic,
        global_dimensions_text=_format_dimensions_text(global_dims),
        topic_dimensions_text=_format_dimensions_text(topic_dims),
        user_query=user_query,
        n_dims=n,
    )

    try:
        raw = call_llm(
            model=SCORE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            json_mode=True,
        )

        from utils import extract_json
        parsed = extract_json(raw)

        # Flexible key lookup — LLM sometimes names fields differently
        desired = parsed.get("desired_vector") or parsed.get("desired")
        priority = parsed.get("priority_vector") or parsed.get("priority")

        if not desired or not priority:
            raise ValueError(f"Missing vectors in response. Keys: {list(parsed.keys())}")

        if len(desired) != n or len(priority) != n:
            raise ValueError(
                f"Vector length mismatch: got {len(desired)}/{len(priority)}, expected {n}"
            )

        # Normalize priority vector so it sums to 1.0
        p_sum = sum(p for p in priority if p > 0)
        if p_sum > 0:
            priority = [p / p_sum if p > 0 else 0.0 for p in priority]

        return desired, priority

    except Exception as e:
        logger.error(f"Query translation failed: {e}. Using uniform fallback.")
        return [-1.0] * n, [1.0 / n] * n


def weighted_distance(
    paper_vector: list[float],
    desired_vector: list[float],
    priority_vector: list[float],
) -> float:
    """Compute priority-weighted distance between a paper and the desired query."""
    total = 0.0
    for pv, dv, pw in zip(paper_vector, desired_vector, priority_vector):
        if dv == -1.0:
            total += SECONDARY_DIM_WEIGHT * abs(pv - 0.5)
        else:
            total += pw * abs(pv - dv)
    return total


def retrieve(
    G: nx.Graph,
    desired_vector: list[float],
    priority_vector: list[float],
    top_k: int = TOP_K_RETRIEVAL,
) -> list[tuple[str, float]]:
    """Retrieve top-k papers by weighted distance (ascending = best)."""
    scored = []
    for node_id, data in G.nodes(data=True):
        if "score_vector" not in data:
            continue
        dist = weighted_distance(data["score_vector"], desired_vector, priority_vector)
        scored.append((node_id, dist))
    return sorted(scored, key=lambda x: x[1])[:top_k]


def retrieve_with_expansion(
    G: nx.Graph,
    desired_vector: list[float],
    priority_vector: list[float],
    top_k_candidates: int = TOP_K_RETRIEVAL,
    top_k_output: int = TOP_K_OUTPUT,
) -> list[dict]:
    """Full retrieval: score → expand via graph neighbors → re-rank → return top-k."""
    candidates = retrieve(G, desired_vector, priority_vector, top_k_candidates)

    if not candidates:
        return []

    # Expand through graph neighborhoods — use ALL ranked candidates as seeds
    expanded_ids = set()
    for paper_id, _ in candidates:
        neighborhood = get_neighborhood(G, paper_id)
        expanded_ids.update(neighborhood)

    # Score all expanded candidates
    all_scored = []
    for node_id in expanded_ids:
        data = G.nodes[node_id]
        if "score_vector" not in data:
            continue
        dist = weighted_distance(data["score_vector"], desired_vector, priority_vector)
        all_scored.append({
            "id": node_id,
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "score_vector": data["score_vector"],
            "distance": dist,
        })

    all_scored.sort(key=lambda x: x["distance"])
    return all_scored[:top_k_output]


def retrieve_on_combined_graph(
    topic_id: int,
    desired_vector: list[float],
    priority_vector: list[float],
    dimensions: list[dict],
    top_k_output: int = TOP_K_OUTPUT,
) -> list[dict]:
    """End-to-end retrieval using the query-adaptive combined graph.

    1. Builds combined graph with priority-weighted dimension edges
    2. Retrieves with expansion
    """
    G = load_combined_graph(topic_id, priority_vector, dimensions)

    if G.number_of_nodes() == 0:
        return []

    return retrieve_with_expansion(
        G, desired_vector, priority_vector,
        top_k_output=top_k_output,
    )
