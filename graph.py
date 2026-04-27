# graph.py — Multi-layer graph construction + traversal for CARTOGRAPH v2
#
# Three edge layers:
#   1. Semantic edges  — SBERT cosine similarity (as before)
#   2. Dimension edges — per-dimension score proximity sub-graphs
#   3. Combined graph  — query-adaptive merge of all layers
import json
import logging

import numpy as np
import networkx as nx

from config import (
    EDGE_SIMILARITY_THRESHOLD,
    DIM_EDGE_THRESHOLD,
    SEMANTIC_EDGE_ALPHA,
    GRAPH_HOP_DEPTH,
    NUM_GLOBAL_DIMENSIONS,
    GLOBAL_DIM_FILTER_THRESHOLD,
)
from db import (
    get_papers_by_topic,
    get_edges_by_topic,
    get_dimension_edges_by_topic,
    insert_edge,
    insert_dimension_edges_bulk,
)
from embed import cosine_similarity

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1: SEMANTIC EDGES (SBERT)
# ══════════════════════════════════════════════════════════════════════════════

def build_semantic_edges(topic_id: int) -> int:
    """Compute pairwise SBERT cosine similarity and store edges above threshold.

    Uses numpy matrix multiplication for vectorized computation.
    Returns the number of edges created.
    """
    papers = [p for p in get_papers_by_topic(topic_id) if p["embedding"]]

    if len(papers) < 2:
        logger.info("Not enough embedded papers to build semantic edges.")
        return 0

    logger.info(f"Computing semantic pairwise similarity for {len(papers)} papers...")

    emb_matrix = np.array([json.loads(p["embedding"]) for p in papers])
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
    emb_normed = emb_matrix / norms
    sim_matrix = emb_normed @ emb_normed.T

    edge_count = 0
    n = len(papers)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= EDGE_SIMILARITY_THRESHOLD:
                insert_edge(papers[i]["id"], papers[j]["id"], sim, "semantic")
                edge_count += 1

    logger.info(f"Created {edge_count} semantic edges (threshold={EDGE_SIMILARITY_THRESHOLD})")
    return edge_count


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2: PER-DIMENSION EDGES (SCORE PROXIMITY)
# ══════════════════════════════════════════════════════════════════════════════

def build_dimension_edges(topic_id: int, dimensions: list[dict]) -> int:
    """Build per-dimension sub-graph edges based on score proximity.

    VECTORIZED: uses numpy broadcasting for ~50-100x speedup over Python loops.
    For each topic-specific dimension, two papers are connected if their
    score proximity (1 - |score_A[d] - score_B[d]|) exceeds DIM_EDGE_THRESHOLD.

    Only topic-specific dimensions get sub-graphs (not the 4 global dims).
    Returns total number of dimension edges created.
    """
    papers = [p for p in get_papers_by_topic(topic_id) if p["score_vector"]]

    if len(papers) < 2:
        logger.info("Not enough scored papers to build dimension edges.")
        return 0

    # Parse score vectors into numpy matrix: shape (N, total_dims)
    paper_ids = [p["id"] for p in papers]
    score_matrix = np.array([json.loads(p["score_vector"]) for p in papers])
    n = len(paper_ids)

    # Only build sub-graphs for topic-specific dimensions (skip first 4 global)
    topic_dims = dimensions[NUM_GLOBAL_DIMENSIONS:]

    logger.info(
        f"Building dimension edges (vectorized): {len(topic_dims)} dimensions × "
        f"{n} papers ({n * (n-1) // 2} pairs each)..."
    )

    total_edges = 0
    bulk_edges = []

    # Upper triangle indices (precompute once)
    tri_i, tri_j = np.triu_indices(n, k=1)

    for dim_idx, dim in enumerate(topic_dims):
        full_idx = NUM_GLOBAL_DIMENSIONS + dim_idx
        dim_name = dim["name"]

        # Vectorized: column of scores for this dimension
        col = score_matrix[:, full_idx]

        # Pairwise proximity matrix via broadcasting
        proximity = 1.0 - np.abs(col[:, None] - col[None, :])

        # Extract upper triangle values that pass threshold
        prox_upper = proximity[tri_i, tri_j]
        mask = prox_upper >= DIM_EDGE_THRESHOLD

        # Build edges for passing pairs
        for idx in np.where(mask)[0]:
            i, j = int(tri_i[idx]), int(tri_j[idx])
            bulk_edges.append(
                (paper_ids[i], paper_ids[j], dim_name, float(prox_upper[idx]))
            )
            total_edges += 1

        # Bulk insert every 5000 edges
        if len(bulk_edges) >= 5000:
            insert_dimension_edges_bulk(bulk_edges)
            bulk_edges = []

    # Flush remaining
    if bulk_edges:
        insert_dimension_edges_bulk(bulk_edges)

    logger.info(f"Created {total_edges} dimension edges across {len(topic_dims)} dimensions")
    return total_edges


def build_all_edges(topic_id: int, dimensions: list[dict]) -> dict:
    """Build both semantic and per-dimension edges.

    Returns dict with edge counts.
    """
    sem = build_semantic_edges(topic_id)
    dim = build_dimension_edges(topic_id, dimensions)
    return {"semantic_edges": sem, "dimension_edges": dim}


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 3: COMBINED GRAPH (QUERY-ADAPTIVE)
# ══════════════════════════════════════════════════════════════════════════════

def load_graph(topic_id: int) -> nx.Graph:
    """Build a basic NetworkX graph from semantic edges only (for compatibility).

    Only includes papers with score_vector.
    """
    G = nx.Graph()

    papers = get_papers_by_topic(topic_id)
    for p in papers:
        if p["score_vector"]:
            G.add_node(
                p["id"],
                title=p["title"],
                abstract=p["abstract"],
                score_vector=json.loads(p["score_vector"]),
                url=p["url"],
            )

    edges = get_edges_by_topic(topic_id)
    for e in edges:
        if G.has_node(e["paper_a"]) and G.has_node(e["paper_b"]):
            G.add_edge(e["paper_a"], e["paper_b"], weight=e["weight"], layer="semantic")

    logger.info(f"Basic graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_combined_graph(
    topic_id: int,
    priority_vector: list[float] | None = None,
    dimensions: list[dict] | None = None,
) -> nx.Graph:
    """Build a query-adaptive combined graph merging semantic + dimension edges.

    Combined edge weight formula:
        W(A,B) = α · W_semantic(A,B) + (1-α) · Σ priority[d] · W_dim_d(A,B)

    Where α = SEMANTIC_EDGE_ALPHA (default 0.3).

    If no priority_vector is given, dimension edges are weighted uniformly.
    """
    G = nx.Graph()

    # Add nodes (only fully scored papers)
    papers = get_papers_by_topic(topic_id)
    for p in papers:
        if p["score_vector"]:
            sv = json.loads(p["score_vector"])

            # Optional: filter by global dimension threshold
            if GLOBAL_DIM_FILTER_THRESHOLD > 0:
                global_scores = sv[:NUM_GLOBAL_DIMENSIONS]
                if any(s < GLOBAL_DIM_FILTER_THRESHOLD for s in global_scores):
                    continue

            G.add_node(
                p["id"],
                title=p["title"],
                abstract=p["abstract"],
                score_vector=sv,
                url=p["url"],
            )

    if G.number_of_nodes() == 0:
        return G

    # ── Collect semantic edge weights ─────────────────────────────────────
    semantic_weights = {}
    for e in get_edges_by_topic(topic_id):
        a, b = e["paper_a"], e["paper_b"]
        if G.has_node(a) and G.has_node(b):
            key = (min(a, b), max(a, b))
            semantic_weights[key] = e["weight"]

    # ── Collect dimension edge weights ────────────────────────────────────
    dim_edge_weights = {}  # (a, b) -> {dim_name: weight}
    for e in get_dimension_edges_by_topic(topic_id):
        a, b = e["paper_a"], e["paper_b"]
        if G.has_node(a) and G.has_node(b):
            key = (min(a, b), max(a, b))
            if key not in dim_edge_weights:
                dim_edge_weights[key] = {}
            dim_edge_weights[key][e["dimension"]] = e["weight"]

    # ── Build priority lookup ─────────────────────────────────────────────
    # Map dimension names to their priority weights
    dim_priority = {}
    if priority_vector and dimensions:
        topic_dims = dimensions[NUM_GLOBAL_DIMENSIONS:]
        for i, dim in enumerate(topic_dims):
            p_idx = NUM_GLOBAL_DIMENSIONS + i
            if p_idx < len(priority_vector):
                dim_priority[dim["name"]] = priority_vector[p_idx]

    # If no priorities given, use uniform weights across topic dims
    if not dim_priority and dimensions:
        topic_dims = dimensions[NUM_GLOBAL_DIMENSIONS:]
        uniform_w = 1.0 / max(len(topic_dims), 1)
        for dim in topic_dims:
            dim_priority[dim["name"]] = uniform_w

    # ── Merge into combined edges ─────────────────────────────────────────
    all_pairs = set(semantic_weights.keys()) | set(dim_edge_weights.keys())
    alpha = SEMANTIC_EDGE_ALPHA

    for pair in all_pairs:
        # Semantic component
        sem_w = semantic_weights.get(pair, 0.0)

        # Dimension component: weighted sum of per-dimension proximities
        dim_w = 0.0
        if pair in dim_edge_weights:
            for dim_name, edge_weight in dim_edge_weights[pair].items():
                p = dim_priority.get(dim_name, 0.0)
                dim_w += p * edge_weight

        # Combined weight
        combined = alpha * sem_w + (1 - alpha) * dim_w

        if combined > 0:
            G.add_edge(
                pair[0], pair[1],
                weight=combined,
                semantic_weight=sem_w,
                dimension_weight=dim_w,
            )

    logger.info(
        f"Combined graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"(α={alpha})"
    )
    return G


# ══════════════════════════════════════════════════════════════════════════════
#  TRAVERSAL
# ══════════════════════════════════════════════════════════════════════════════

def get_neighborhood(G: nx.Graph, node_id: str, depth: int = GRAPH_HOP_DEPTH) -> set[str]:
    """BFS up to `depth` hops from node_id. Returns set of node IDs."""
    if node_id not in G:
        return set()

    visited = {node_id}
    frontier = {node_id}
    for _ in range(depth):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(G.neighbors(n))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
    return visited
