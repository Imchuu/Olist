"""Configuration settings for Flow 2 LLM customer satisfaction prediction."""

from pathlib import Path


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

def get_input_path() -> Path:
    """Return the path to the input dataset.

    Returns:
        Path to the order-level CSV dataset.
    """
    return Path("ML/data/olist_sampled_10k.csv")


def get_output_path() -> Path:
    """Return the path where results will be saved.

    Returns:
        Path for the output CSV (customer_id, P_LLM).
    """
    return Path("ML/results/llm_predictions.csv")


def get_phase1_output_path() -> Path:
    """Return the output path for Phase 1 narrative-only mode.

    Returns:
        Path for a CSV containing customer_id and profile_text.
    """
    return Path("ML/results/phase1_narratives.csv")


# -----------------------------------------------------------------------------
# Local LLM API (LM Studio - OpenAI compatible)
# -----------------------------------------------------------------------------

def get_local_api_base_url() -> str:
    """Return the base URL for local LM Studio server.

    Returns:
        URL string pointing to LM Studio OpenAI-compatible API.
    """
    return "http://localhost:1234/v1"


def get_local_model() -> str:
    """Return a single model alias used for all LLM calls.

    Returns:
        Local model identifier.
    """
    return "local-model"


def get_narrative_model() -> str:
    """Return model used in Phase 1 narrative generation.

    Returns:
        Local model identifier.
    """
    return get_local_model()


def get_prediction_model() -> str:
    """Return model used in Phase 2 satisfaction prediction.

    Returns:
        Local model identifier.
    """
    return get_local_model()


def get_llm_model() -> str:
    """Return the default model name (alias for get_prediction_model).

    Returns:
        Local model identifier.
    """
    return get_prediction_model()


def get_llm_temperature() -> float:
    """Return the LLM temperature. 0.0 for deterministic inference.

    Returns:
        Float temperature value.
    """
    return 0.0


def get_request_timeout() -> int:
    """Return HTTP timeout in seconds for local API calls.

    Returns:
        Timeout in seconds.
    """
    return 120


def get_stop_after_phase1() -> bool:
    """Return whether pipeline should stop after Phase 1 output.

    Returns:
        True to generate narratives then stop.
    """
    return True


def get_fail_fast_on_llm_error() -> bool:
    """Return whether pipeline should stop immediately on LLM failures.

    Returns:
        True to stop immediately when local LLM call fails.
    """
    return True
