"""Phase 1: Generate natural language customer narratives via LLM."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Callable, Optional

import pandas as pd

from .config import (
    FAIL_FAST_ON_LLM_ERROR,
    LLM_MODEL_PHASE1,
    LLM_TEMPERATURE,
    MAX_WORKERS,
    PHASE1_MAX_TOKENS,
    PROGRESS_LOG_EVERY_N,
)
from .llm_predictor import call_llm
from .prompt_builder import build_narrative_system_prompt, build_narrative_user_prompt
from .utils import ProgressBar

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
    model = model or LLM_MODEL_PHASE1
    system_prompt = build_narrative_system_prompt()
    user_prompt = build_narrative_user_prompt(features, customer_id=customer_id)

    try:
        narrative = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=LLM_TEMPERATURE,
            max_tokens=PHASE1_MAX_TOKENS,
        )
        logger.debug("Narrative generated for customer %s (%d chars)", customer_id, len(narrative))
        return narrative
    except Exception as exc:  # noqa: BLE001
        logger.error("Narrative generation failed for customer %s: %s", customer_id, exc)
        if FAIL_FAST_ON_LLM_ERROR:
            raise ConnectionError(
                "Phase 1 stopped due to LLM call failure. "
                f"Please ensure LM Studio local server is running and model '{LLM_MODEL_PHASE1}' is loaded."
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
    existing_profiles: Optional[dict[str, str]] = None,
    on_profile_generated: Optional[Callable[[str, str], None]] = None,
) -> pd.DataFrame:
    """Generate narrative profiles for all customers in the DataFrame.

    Adds a 'profile_text' column to the input DataFrame.

    Args:
        customers_df: Customer-level DataFrame from feature_aggregator.
        model: Ollama model override for Phase 1.
        existing_profiles: Optional map of customer_id -> profile_text for resume.
        on_profile_generated: Optional callback called after generating each new profile.

    Returns:
        Copy of the DataFrame with an added 'profile_text' column.
    """
    total = len(customers_df)
    progress_every = PROGRESS_LOG_EVERY_N
    logger.info(
        "Generating narratives for %d customers (Phase 1) with %d worker(s)...",
        total,
        MAX_WORKERS,
    )
    progress_bar = ProgressBar(total=total, label="Phase 1")

    rows = customers_df.to_dict("records")
    narratives: list[str] = [""] * total
    cached = existing_profiles or {}
    cached_count = 0

    def _build_one(idx: int, row_dict: dict) -> tuple[int, str]:
        cid = str(row_dict.get("customer_id", idx))
        features = {k: row_dict[k] for k in _FEATURE_KEYS if k in row_dict}
        narrative = generate_customer_narrative(features, customer_id=cid, model=model)
        return idx, narrative

    completed = 0
    if MAX_WORKERS <= 1:
        for i, row_dict in enumerate(rows):
            cid = str(row_dict.get("customer_id", i))
            if cid in cached:
                narratives[i] = str(cached[cid])
                cached_count += 1
            else:
                idx, narrative = _build_one(i, row_dict)
                narratives[idx] = narrative
                if on_profile_generated:
                    on_profile_generated(cid, narrative)
            completed += 1
            progress_bar.update(completed)
            if completed % progress_every == 0 or completed == total:
                logger.info("  Phase 1 progress: %d / %d", completed, total)
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for i, row_dict in enumerate(rows):
                cid = str(row_dict.get("customer_id", i))
                if cid in cached:
                    narratives[i] = str(cached[cid])
                    cached_count += 1
                    completed += 1
                    progress_bar.update(completed)
                    if completed % progress_every == 0 or completed == total:
                        logger.info("  Phase 1 progress: %d / %d", completed, total)
                    continue
                future = executor.submit(_build_one, i, row_dict)
                futures[future] = cid

            for future in as_completed(futures):
                idx, narrative = future.result()
                narratives[idx] = narrative
                cid = futures[future]
                if on_profile_generated:
                    on_profile_generated(cid, narrative)
                completed += 1
                progress_bar.update(completed)
                if completed % progress_every == 0 or completed == total:
                    logger.info("  Phase 1 progress: %d / %d", completed, total)

    progress_bar.finish()
    if cached_count > 0:
        logger.info("Phase 1 resumed %d existing narratives from checkpoint.", cached_count)

    result = customers_df.copy()
    result[_NARRATIVE_COL] = narratives
    logger.info("Phase 1 complete. Narratives added to DataFrame.")
    return result
