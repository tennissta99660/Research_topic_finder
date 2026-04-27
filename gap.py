# gap.py — Gap detection + research direction synthesis for CARTOGRAPH v2
#
# Gap analysis now reports on both global and topic-specific dimensions separately.
import json
import logging
import networkx as nx

from config import (
    SYNTHESIS_MODEL,
    GAP_MIN_NEIGHBOR_DIST,
    NUM_GLOBAL_DIMENSIONS,
)
from llm import call_llm

logger = logging.getLogger(__name__)


SYNTHESIS_PROMPT = """
You are a research advisor in the field of {topic}.

A researcher is looking for: "{user_query}"

The most relevant existing papers are:
{paper_summaries}

Analysis of the research space reveals these gaps — areas where the researcher's
desired profile differs significantly from what currently exists:

=== GLOBAL QUALITY GAPS ===
{global_gap_description}

=== TOPIC-SPECIFIC TECHNICAL GAPS ===
{topic_gap_description}

Based on this, synthesize 2–3 concrete, novel research directions that:
1. Build on the existing papers above
2. Address the identified gaps (especially the topic-specific technical ones)
3. Are specific enough to be actionable (not just "combine X and Y")
4. Are realistic for a small research team

For each direction, provide:
- A one-line title
- A 2–3 sentence description of the core idea
- Why it fills the gap
- A suggested first experiment

Return as a JSON array of direction objects with keys: "title", "description", "gap_rationale", "first_experiment".
"""


def detect_gap(
    G: nx.Graph,
    top_papers: list[dict],
    dimensions: list[dict],
    desired_vector: list[float],
) -> dict:
    """Detect sparse regions in score-space by comparing desired vs available.

    Returns gap info split into global_gaps and topic_gaps.
    """
    if not top_papers:
        return {"global_gaps": [], "topic_gaps": [], "mean_vector": []}

    vectors = [p["score_vector"] for p in top_papers]
    n_dims = len(vectors[0])
    mean_vec = [
        sum(v[i] for v in vectors) / len(vectors) for i in range(n_dims)
    ]

    global_gaps = []
    topic_gaps = []

    for i, dim in enumerate(dimensions):
        if i >= len(desired_vector) or desired_vector[i] == -1.0:
            continue
        gap_size = abs(desired_vector[i] - mean_vec[i])
        if gap_size > GAP_MIN_NEIGHBOR_DIST:
            gap_entry = {
                "dimension": dim["name"],
                "description": dim["description"],
                "desired": round(desired_vector[i], 3),
                "available": round(mean_vec[i], 3),
                "gap": round(gap_size, 3),
            }
            if i < NUM_GLOBAL_DIMENSIONS:
                global_gaps.append(gap_entry)
            else:
                topic_gaps.append(gap_entry)

    global_gaps.sort(key=lambda x: x["gap"], reverse=True)
    topic_gaps.sort(key=lambda x: x["gap"], reverse=True)

    return {
        "global_gaps": global_gaps,
        "topic_gaps": topic_gaps,
        "mean_vector": mean_vec,
    }


def _format_paper_summaries(papers: list[dict]) -> str:
    """Format top papers for the synthesis prompt."""
    lines = []
    for i, p in enumerate(papers, 1):
        scores_str = ", ".join(f"{s:.2f}" for s in p["score_vector"][:10])
        if len(p["score_vector"]) > 10:
            scores_str += f", ... ({len(p['score_vector'])} total)"
        lines.append(
            f"{i}. \"{p['title']}\" (distance={p['distance']:.3f})\n"
            f"   Scores: [{scores_str}]"
        )
    return "\n".join(lines)


def _format_gap_list(gaps: list[dict]) -> str:
    """Format a list of gaps for the synthesis prompt."""
    if not gaps:
        return "No significant gaps detected in this tier."
    lines = []
    for g in gaps:
        lines.append(
            f"  - {g['dimension']}: desired={g['desired']}, available={g['available']} "
            f"(gap={g['gap']}). {g['description']}"
        )
    return "\n".join(lines)


def synthesize_directions(
    topic: str,
    user_query: str,
    top_papers: list[dict],
    gap_info: dict,
    dimensions: list[dict],
) -> list[dict]:
    """Use LLM to synthesize concrete research directions from retrieved papers + gap analysis."""
    prompt = SYNTHESIS_PROMPT.format(
        topic=topic,
        user_query=user_query,
        paper_summaries=_format_paper_summaries(top_papers),
        global_gap_description=_format_gap_list(gap_info.get("global_gaps", [])),
        topic_gap_description=_format_gap_list(gap_info.get("topic_gaps", [])),
    )

    try:
        raw = call_llm(
            model=SYNTHESIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            json_mode=True,
        )

        from utils import extract_json
        parsed = extract_json(raw)

        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            for key in parsed:
                if isinstance(parsed[key], list):
                    return parsed[key]
            return [parsed]
        else:
            logger.warning(f"Unexpected synthesis response type: {type(parsed)}")
            return []

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return []
