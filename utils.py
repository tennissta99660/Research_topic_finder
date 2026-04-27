# utils.py — Shared utilities for CARTOGRAPH v2
import json
import re


def extract_json(text: str):
    """Extract a JSON object or array from LLM output that may contain
    markdown fences, preamble text, or other non-JSON content.

    Handles common LLM output patterns:
      - Pure JSON
      - ```json ... ``` fenced blocks
      - "Here is the result:\n\n{...}"
      - Mixed text with embedded JSON

    Returns the parsed Python object (dict or list).
    Raises ValueError if no valid JSON is found.
    """
    if not text or not text.strip():
        raise ValueError("Empty response — LLM returned no content")

    # 0. Strip <think>...</think> tags from "thinking" models (Qwen3, etc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if not text:
        raise ValueError("LLM response was only thinking tags — no actual content")

    # 1. Try parsing the whole thing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Try extracting from ```json ... ``` fenced blocks
    fenced = re.findall(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    for block in fenced:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    # 3. Find the first { or [ and try to parse from there
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find the matching closing bracket by scanning from the end
        end = text.rfind(end_char)
        if end == -1 or end <= start:
            continue
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No valid JSON found in LLM response: {text[:300]}")


def format_dimensions_text(dimensions: list[dict]) -> str:
    """Format a list of dimensions into numbered text for LLM prompts.

    Used by score.py, query.py, and any module building dimension-aware prompts.
    """
    lines = []
    for i, dim in enumerate(dimensions, 1):
        lines.append(
            f"{i}. {dim['name']}: {dim['description']} "
            f"(0={dim['low']}, 1={dim['high']})"
        )
    return "\n".join(lines)
