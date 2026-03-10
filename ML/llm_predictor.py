"""Call local LM Studio API for Phase 1 and Phase 2 inference."""

import logging
import time
from typing import Optional

import requests

from .config import (
    get_local_api_base_url,
    get_llm_model,
    get_llm_temperature,
    get_request_timeout,
)

logger = logging.getLogger(__name__)

_CHAT_ENDPOINT = "/chat/completions"
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds between retries


def check_local_llm_connection(required_model: Optional[str] = None) -> bool:
    """Check whether LM Studio local API is reachable and model is available.

    Args:
        required_model: Optional model name that must exist in /models list.

    Returns:
        True if local API responds and (if provided) model is available.
    """
    base_url = get_local_api_base_url()
    models_url = f"{base_url}/models"
    try:
        response = requests.get(models_url, timeout=5)
        response.raise_for_status()
        if required_model:
            payload = response.json()
            model_items = payload.get("data", []) if isinstance(payload, dict) else []
            available_models = {
                str(item.get("id", "")).strip() for item in model_items if isinstance(item, dict)
            }
            if required_model not in available_models:
                logger.error(
                    "LM Studio is reachable but model '%s' is not loaded. Available: %s",
                    required_model,
                    sorted(m for m in available_models if m),
                )
                return False
        return True
    except requests.exceptions.RequestException as exc:
        logger.error("LM Studio connection check failed: %s", exc)
        return False


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """Send a prompt pair to local LM Studio API and return response text.

    Uses the OpenAI-compatible /chat/completions endpoint.
    Retries up to _MAX_RETRIES times on transient failures.

    Args:
        system_prompt: Sets the model's role and output format.
        user_prompt: The customer profile or narrative to process.
        model: Local model name; defaults to config value.
        temperature: Generation temperature; 0.0 for deterministic output.

    Returns:
        Raw text content from the model's response.

    Raises:
        ConnectionError: If local server is unreachable after all retries.
        ValueError: If the response is malformed or empty.
    """
    model = model or get_llm_model()
    temperature = temperature if temperature is not None else get_llm_temperature()
    base_url = get_local_api_base_url()
    url = f"{base_url}{_CHAT_ENDPOINT}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    last_exc: Exception = RuntimeError("No attempts made.")
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.debug("LLM call attempt %d/%d  model=%s", attempt, _MAX_RETRIES, model)
            response = requests.post(url, json=payload, timeout=get_request_timeout())
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise ValueError("LM Studio returned an empty response.")
            logger.debug("LLM response received (%d chars)", len(content))
            return content.strip()
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            logger.warning("LM Studio unreachable (attempt %d): %s", attempt, exc)
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            logger.warning("HTTP error (attempt %d): %s", attempt, exc)
        except (KeyError, ValueError) as exc:
            last_exc = exc
            logger.warning("Bad response format (attempt %d): %s", attempt, exc)

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY)

    raise ConnectionError(
        f"LM Studio API failed after {_MAX_RETRIES} attempts. Last error: {last_exc}"
    )


def call_llm_batch(
    prompt_pairs: list[tuple[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> list[str]:
    """Call the LLM sequentially for a list of (system, user) prompt pairs.

    LM Studio local endpoint is called sequentially for deterministic behavior.

    Args:
        prompt_pairs: List of (system_prompt, user_prompt) tuples.
        model: Local model name override.
        temperature: Temperature override.

    Returns:
        List of raw response strings, same order as prompt_pairs.
        On individual failure, inserts an empty string and logs the error.
    """
    results: list[str] = []
    for i, (sys_prompt, usr_prompt) in enumerate(prompt_pairs):
        try:
            text = call_llm(sys_prompt, usr_prompt, model=model, temperature=temperature)
        except Exception as exc:  # noqa: BLE001
            logger.error("Batch item %d failed: %s", i, exc)
            text = ""
        results.append(text)
    return results
