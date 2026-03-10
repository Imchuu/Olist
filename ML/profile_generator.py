"""Phase 1: Generate natural language customer narratives via LLM."""

import logging
from typing import Optional

import pandas as pd

from .config import get_narrative_model, get_llm_temperature, get_fail_fast_on_llm_error
from .llm_predictor import call_llm
from .prompt_builder import build_narrative_system_prompt, build_narrative_user_prompt

logger = logging.getLogger(__name__)

_NARRATIVE_COL = "profile_text"

_FEATURE_KEYS = [
    "num_orders",
    "avg_price",
    "avg_freight",
    "total_spend",
    "avg_delivery_time",
    "late_delivery_ratio",
    "max_delay_days",
    "num_product_categories",
    "num_payment_types",
]


def generate_customer_narrative(
    features: dict,
    customer_id: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Generate a structured natural language narrative for one customer.

    Calls the Phase 1 LLM with structured features and returns the narrative
    that will be fed into Phase 2 for probability estimation.

    Args:
        features: Dict of aggregated customer metrics (output of feature_aggregator).
        customer_id: Optional ID used for tracing in the prompt.
        model: Ollama model override for Phase 1.

    Returns:
        Narrative text string, or a fallback message on failure.
    """
    model = model or get_narrative_model()
    system_prompt = build_narrative_system_prompt()
    user_prompt = build_narrative_user_prompt(features, customer_id=customer_id)

    try:
        narrative = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=get_llm_temperature(),
        )
        logger.debug("Narrative generated for customer %s (%d chars)", customer_id, len(narrative))
        return narrative
    except Exception as exc:  # noqa: BLE001
        logger.error("Narrative generation failed for customer %s: %s", customer_id, exc)
        if get_fail_fast_on_llm_error():
            raise ConnectionError(
                "Phase 1 stopped due to LLM call failure. "
                "Please ensure LM Studio local server is running and model 'local-model' is loaded."
            ) from exc
        # Return a minimal fallback narrative so Phase 2 can still proceed
        return (
            f"Customer Summary:\n"
            f"Customer placed {features.get('num_orders', '?')} order(s) "
            f"with a late delivery ratio of {features.get('late_delivery_ratio', '?')}.\n\n"
            f"Key Positive Signals:\n* None\n\n"
            f"Key Negative Signals:\n* Unable to generate detailed narrative.\n\n"
            f"Notable Events:\n* None"
        )


def generate_all_profiles(
    customers_df: pd.DataFrame,
    model: Optional[str] = None,
) -> pd.DataFrame:
    """Generate narrative profiles for all customers in the DataFrame.

    Adds a 'profile_text' column to the input DataFrame.

    Args:
        customers_df: Customer-level DataFrame from feature_aggregator.
        model: Ollama model override for Phase 1.

    Returns:
        Copy of the DataFrame with an added 'profile_text' column.
    """
    total = len(customers_df)
    logger.info("Generating narratives for %d customers (Phase 1)...", total)

    narratives: list[str] = []
    for i, row in customers_df.iterrows():
        cid = str(row.get("customer_id", i))
        features = {k: row[k] for k in _FEATURE_KEYS if k in row}
        narrative = generate_customer_narrative(features, customer_id=cid, model=model)
        narratives.append(narrative)

        if (i + 1) % 50 == 0 or (i + 1) == total:  # type: ignore[operator]
            logger.info("  Phase 1 progress: %d / %d", i + 1, total)  # type: ignore[operator]

    result = customers_df.copy()
    result[_NARRATIVE_COL] = narratives
    logger.info("Phase 1 complete. Narratives added to DataFrame.")
    return result
