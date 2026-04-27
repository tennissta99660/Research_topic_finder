# score.py — LLM scoring of papers across all dimensions for CARTOGRAPH v2
#
# Scores papers on both global (4) and topic-specific dimensions in a single call.
# Handles up to 50 dimensions per paper.
import json
import time
import logging

from tqdm import tqdm

from config import SCORE_MODEL, NUM_GLOBAL_DIMENSIONS
from llm import call_llm
from db import get_unscored_papers, update_score_vector
from utils import format_dimensions_text as _format_dimensions_text

logger = logging.getLogger(__name__)


SCORING_PROMPT = """
You are a research evaluator. Score the following paper on each dimension from 0.0 to 1.0.

Paper title: {title}
Abstract: {abstract}

=== GLOBAL DIMENSIONS (universal research quality axes) ===
{global_dimensions_text}

=== TOPIC-SPECIFIC DIMENSIONS (field-specific technical axes) ===
{topic_dimensions_text}

Scoring rules:
- Be critical and precise. Use the full range [0.0, 1.0].
- Score based only on what the abstract explicitly states.
- Do not infer unstated capabilities.
- Score ALL dimensions listed above (both global and topic-specific).

Return ONLY a JSON object with dimension names as keys and float scores as values.
No preamble, no explanation.
Example: {{"novelty": 0.6, "rigor": 0.8, "scalability": 0.3, "heterogeneity": 0.7}}
"""


def _parse_score_response(
    raw: str,
    dimensions: list[dict],
) -> list[float] | None:
    """Parse LLM JSON response into an ordered score vector.

    Returns None if parsing fails or dimension names don't match.
    """
    try:
        from utils import extract_json
        scores = extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(scores, dict):
        return None

    dim_names = [d["name"] for d in dimensions]
    vector = []
    for name in dim_names:
        if name not in scores:
            logger.warning(f"Missing dimension '{name}' in LLM response")
            return None
        val = scores[name]
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        # Clamp to [0.0, 1.0]
        val = max(0.0, min(1.0, val))
        vector.append(val)

    return vector


def score_papers(
    topic_id: int,
    dimensions: list[dict],
    batch_size: int = 10,
) -> int:
    """Score all unscored papers for a topic across all dimensions (global + topic).

    Processes in batches with rate-limit sleep between batches.
    Returns the number of papers successfully scored.
    """
    papers = get_unscored_papers(topic_id)

    if not papers:
        logger.info("No unscored papers found.")
        return 0

    # Split dimensions into global and topic for separate prompt sections
    global_dims = dimensions[:NUM_GLOBAL_DIMENSIONS]
    topic_dims = dimensions[NUM_GLOBAL_DIMENSIONS:]

    global_text = _format_dimensions_text(global_dims)
    topic_text = _format_dimensions_text(topic_dims)

    scored_count = 0

    for i in tqdm(range(0, len(papers), batch_size), desc="Scoring papers"):
        batch = papers[i : i + batch_size]

        for paper in batch:
            prompt = SCORING_PROMPT.format(
                title=paper["title"],
                abstract=paper["abstract"],
                global_dimensions_text=global_text,
                topic_dimensions_text=topic_text,
            )

            # Per-paper retry: up to 3 attempts before skipping
            max_paper_retries = 3
            for attempt in range(1, max_paper_retries + 1):
                try:
                    raw = call_llm(
                        model=SCORE_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        json_mode=True,
                    )
                    vector = _parse_score_response(raw, dimensions)

                    if vector is None:
                        logger.warning(
                            f"Malformed score for '{paper['title'][:50]}' "
                            f"(id={paper['id']}, attempt {attempt}/{max_paper_retries}). "
                            f"Raw: {raw[:200]}"
                        )
                        if attempt < max_paper_retries:
                            time.sleep(0.5)
                            continue
                        break

                    update_score_vector(paper["id"], vector)
                    scored_count += 1
                    break  # success

                except Exception as e:
                    logger.error(
                        f"Error scoring '{paper['title'][:50]}' "
                        f"(id={paper['id']}, attempt {attempt}/{max_paper_retries}): {e}"
                    )
                    if attempt < max_paper_retries:
                        time.sleep(1)
                    continue

        # Rate-limit sleep between batches
        if i + batch_size < len(papers):
            time.sleep(2)

    logger.info(f"Scored {scored_count}/{len(papers)} papers")
    return scored_count
