# dimensions.py — Two-tier dimension generation + caching for CARTOGRAPH v2
#
# Tier 1: 4 Global dimensions (novelty, rigor, impact, reproducibility)
#          — hardcoded, universal across all topics
# Tier 2: N Topic-specific dimensions (user-configurable, up to 46)
#          — LLM-generated, cached per topic
import json
import os
import logging

from config import (
    SCORE_MODEL,
    MAX_TOPIC_DIMENSIONS,
    MIN_TOPIC_DIMENSIONS,
    DEFAULT_TOPIC_DIMENSIONS,
    GLOBAL_DIMENSIONS,
    DIMENSIONS_DIR,
)

logger = logging.getLogger(__name__)


DIMENSION_PROMPT = """
You are a research cartographer specializing in the field of {topic}.

Generate exactly {n} dimensions that together characterize the TECHNICAL aspects of
any research paper in this field. These dimensions must be SPECIFIC to {topic}.

IMPORTANT: Do NOT include generic research quality dimensions like novelty, rigor,
impact, or reproducibility — those are handled separately as global axes.
Focus entirely on field-specific technical characteristics.

Requirements for each dimension:
1. Specific to {topic} — not generic across all research fields
2. Independently scoreable from a paper abstract alone (no need to read full paper)
3. Orthogonal to all other dimensions (minimal conceptual overlap)
4. Has a clear, concrete low-end (0.0) and high-end (1.0) interpretation
5. Covers a meaningful technical aspect that varies substantially across papers

Mix methodology dimensions (approach type, technique specifics) with scope dimensions
(scale, breadth of application, domain specificity).

Return ONLY a JSON array with no preamble or explanation:
[
  {{
    "name": "short_identifier",
    "description": "one sentence describing what this measures",
    "low": "concrete description of what score 0.0 looks like",
    "high": "concrete description of what score 1.0 looks like"
  }}
]
"""


def _topic_slug(topic: str) -> str:
    """Convert topic name to a filesystem-safe slug.

    'Graph Neural Networks' -> 'graph_neural_networks'
    """
    return topic.lower().strip().replace(" ", "_")


def _cache_path(topic: str) -> str:
    """Return the full path to the cached dimension JSON file."""
    return os.path.join(DIMENSIONS_DIR, f"{_topic_slug(topic)}.json")


def _validate_topic_dimensions(dims: list[dict], expected_count: int) -> None:
    """Raise ValueError if topic-specific dimensions don't meet requirements."""
    if not isinstance(dims, list):
        raise ValueError("Dimensions must be a JSON array")
    if len(dims) < MIN_TOPIC_DIMENSIONS:
        raise ValueError(
            f"Expected at least {MIN_TOPIC_DIMENSIONS} topic dimensions, got {len(dims)}"
        )
    if len(dims) > MAX_TOPIC_DIMENSIONS:
        raise ValueError(
            f"Expected at most {MAX_TOPIC_DIMENSIONS} topic dimensions, got {len(dims)}"
        )
    required_keys = {"name", "description", "low", "high"}
    names_seen = set()
    # Also check against global dimension names
    global_names = {d["name"] for d in GLOBAL_DIMENSIONS}
    for i, dim in enumerate(dims):
        missing = required_keys - set(dim.keys())
        if missing:
            raise ValueError(f"Dimension {i} missing keys: {missing}")
        name = dim["name"]
        if name in names_seen:
            raise ValueError(f"Duplicate dimension name: {name}")
        if name in global_names:
            raise ValueError(
                f"Topic dimension '{name}' conflicts with global dimension. "
                "Remove it — global dims are added automatically."
            )
        names_seen.add(name)


def generate_topic_dimensions(topic: str, n: int = DEFAULT_TOPIC_DIMENSIONS) -> list[dict]:
    """Call the LLM to generate field-specific scoring dimensions (topic tier only)."""
    from llm import call_llm
    from config import SCORE_MODEL
    from utils import extract_json

    prompt = DIMENSION_PROMPT.format(topic=topic, n=n)

    raw = call_llm(
        model=SCORE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        json_mode=True,
    )

    try:
        parsed = extract_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:500]}")

    # The LLM may wrap the array in an object key — handle both cases
    if isinstance(parsed, dict):
        for key in parsed:
            if isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            raise ValueError(f"LLM returned a dict with no array value: {list(parsed.keys())}")

    # Trim to requested count if LLM returned extra
    if len(parsed) > n:
        parsed = parsed[:n]

    _validate_topic_dimensions(parsed, n)
    return parsed


def get_global_dimensions() -> list[dict]:
    """Return the 4 hardcoded global dimensions (always the same)."""
    return list(GLOBAL_DIMENSIONS)


def get_topic_dimensions(topic: str, n: int = DEFAULT_TOPIC_DIMENSIONS) -> list[dict]:
    """Load topic-specific dimensions from cache if they exist, otherwise generate and cache.

    CRITICAL: Never regenerate dimensions for an existing topic.
    """
    cache_file = _cache_path(topic)

    if os.path.exists(cache_file):
        logger.info(f"Loading cached topic dimensions for '{topic}'")
        with open(cache_file, "r", encoding="utf-8") as f:
            dims = json.load(f)
        return dims

    logger.info(f"Generating {n} topic dimensions for '{topic}'")
    dims = generate_topic_dimensions(topic, n)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(dims, f, indent=2)
    logger.info(f"Cached {len(dims)} topic dimensions to {cache_file}")

    return dims


def get_all_dimensions(topic: str, n_topic: int = DEFAULT_TOPIC_DIMENSIONS) -> list[dict]:
    """Return the full dimension set: global (4) + topic-specific (n_topic).

    The returned list is always ordered: [global_0, ..., global_3, topic_0, ..., topic_N].
    This ordering must NEVER change — score vectors depend on it.
    """
    global_dims = get_global_dimensions()
    topic_dims = get_topic_dimensions(topic, n_topic)
    return global_dims + topic_dims


def get_dimension_names(dimensions: list[dict]) -> list[str]:
    """Extract ordered list of dimension names."""
    return [d["name"] for d in dimensions]


def is_global_dimension(dim_name: str) -> bool:
    """Check if a dimension name belongs to the global tier (case-insensitive)."""
    return dim_name.lower() in {d["name"].lower() for d in GLOBAL_DIMENSIONS}
