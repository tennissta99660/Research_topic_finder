# llm.py — LLM client for CARTOGRAPH
#
# Priority chain: Lightning AI (remote Ollama) → Local Ollama → Groq cloud
# Automatically falls through to the next backend on failure.
import logging
import time
import json
import httpx

from openai import OpenAI

from config import (
    LIGHTNING_OLLAMA_URL,
    LIGHTNING_MODEL,
    LOCAL_OLLAMA_URL,
    LOCAL_MODEL,
    GROQ_API_KEYS,
)

logger = logging.getLogger(__name__)

# ── Groq key rotation state ──────────────────────────────────────────────────
_current_key_index = 0
_dead_keys: set[int] = set()


def _get_live_groq_indices() -> list[int]:
    return [i for i in range(len(GROQ_API_KEYS)) if i not in _dead_keys]


# ── Backend callers ──────────────────────────────────────────────────────────

def _call_ollama(
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    json_mode: bool,
    timeout_seconds: float = 600.0,
) -> str:
    """Call an Ollama-compatible endpoint (local or remote Lightning AI)."""
    client = OpenAI(
        base_url=base_url,
        api_key="ollama",
        timeout=httpx.Timeout(timeout_seconds, connect=30.0),
    )

    # Disable thinking/reasoning for Qwen3 models — saves tokens & latency
    # Append /no_think to the system message so the model skips <think> blocks
    patched_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            patched_messages.append({
                **msg,
                "content": msg["content"] + "\n\n/no_think",
            })
        else:
            patched_messages.append(msg)

    kwargs = {
        "model": model,
        "messages": patched_messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def _call_groq(
    model: str,
    messages: list[dict],
    temperature: float,
    json_mode: bool,
    max_retries: int,
) -> str:
    """Call Groq cloud with key rotation on rate limits."""
    from groq import Groq

    global _current_key_index
    keys = GROQ_API_KEYS
    if not keys:
        raise ValueError("No GROQ API keys configured.")

    live_indices = _get_live_groq_indices()
    if not live_indices:
        raise ValueError("All Groq API keys are dead (401/403).")

    total_attempts = len(live_indices) * max_retries
    last_error = None

    for attempt in range(total_attempts):
        live_indices = _get_live_groq_indices()
        if not live_indices:
            break

        key_idx = live_indices[_current_key_index % len(live_indices)]
        key = keys[key_idx]
        key_label = f"key_{key_idx + 1}"

        try:
            client = Groq(api_key=key, max_retries=0)
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            if any(p in error_str for p in ["401", "403", "unauthorized", "forbidden", "invalid_api_key"]):
                _dead_keys.add(key_idx)
                logger.warning(f"⚠️ {key_label} is DEAD, removed. Live: {len(_get_live_groq_indices())}")
                continue

            if any(p in error_str for p in ["429", "rate_limit", "rate limit", "too many"]):
                _current_key_index += 1
                logger.warning(f"Rate limited {key_label}, rotating... (attempt {attempt+1}/{total_attempts})")
                time.sleep(0.5)
            else:
                logger.error(f"Groq error on {key_label}: {e}")
                _current_key_index += 1
                time.sleep(1)

    raise RuntimeError(f"All Groq keys exhausted. Last error: {last_error}")


# ── Main entry point ─────────────────────────────────────────────────────────

def call_llm(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    json_mode: bool = True,
    max_retries: int = 3,
) -> str:
    """Call LLM with automatic fallback chain:

    1. Lightning AI (remote Ollama on cloud GPU) — if URL configured
    2. Local Ollama — if running
    3. Groq cloud — last resort

    The `model` parameter is only used for the Groq fallback.
    """
    errors = []

    # ── 1. Try Lightning AI (primary) ────────────────────────────────────────
    if LIGHTNING_OLLAMA_URL and LIGHTNING_OLLAMA_URL.strip():
        try:
            logger.info(f"⚡ Calling Lightning AI ({LIGHTNING_MODEL})...")
            result = _call_ollama(
                base_url=LIGHTNING_OLLAMA_URL.strip(),
                model=LIGHTNING_MODEL,
                messages=messages,
                temperature=temperature,
                json_mode=json_mode,
                timeout_seconds=300.0,  # 5 min — generous for cloud
            )
            logger.info("⚡ Lightning AI responded successfully.")
            return result
        except Exception as e:
            errors.append(f"Lightning AI: {e}")
            logger.warning(f"⚡ Lightning AI failed: {e} — falling back...")

    # ── 2. Try Local Ollama (fallback) ───────────────────────────────────────
    try:
        logger.info(f"🏠 Calling Local Ollama ({LOCAL_MODEL})...")
        result = _call_ollama(
            base_url=LOCAL_OLLAMA_URL,
            model=LOCAL_MODEL,
            messages=messages,
            temperature=temperature,
            json_mode=json_mode,
            timeout_seconds=600.0,  # 10 min — local GPU can be slow
        )
        logger.info("🏠 Local Ollama responded successfully.")
        return result
    except Exception as e:
        errors.append(f"Local Ollama: {e}")
        error_str = str(e).lower()
        if "connection" in error_str or "refused" in error_str:
            logger.warning("🏠 Local Ollama not running — falling back to Groq...")
        elif "not found" in error_str or "404" in error_str:
            logger.warning(f"🏠 Model '{LOCAL_MODEL}' not found locally — falling back to Groq...")
        else:
            logger.warning(f"🏠 Local Ollama failed: {e} — falling back to Groq...")

    # ── 3. Try Groq cloud (last resort) ──────────────────────────────────────
    if GROQ_API_KEYS:
        try:
            logger.info(f"☁️ Calling Groq cloud ({model})...")
            result = _call_groq(model, messages, temperature, json_mode, max_retries)
            logger.info("☁️ Groq responded successfully.")
            return result
        except Exception as e:
            errors.append(f"Groq: {e}")
            logger.error(f"☁️ Groq also failed: {e}")

    # ── All backends failed ──────────────────────────────────────────────────
    error_summary = "\n  ".join(errors) if errors else "No backends available"
    raise RuntimeError(
        f"All LLM backends failed:\n  {error_summary}\n\n"
        f"To fix:\n"
        f"  1. Set LIGHTNING_OLLAMA_URL in .env (primary)\n"
        f"  2. Start local Ollama: ollama serve (fallback)\n"
        f"  3. Add GROQ_API_KEY_* to .env (last resort)"
    )
