"""Build LLM prompts for Phase 1 (narrative) and Phase 2 (prediction)."""

from typing import Optional


# ---------------------------------------------------------------------------
# Phase 1 – Narrative generation
# ---------------------------------------------------------------------------

def build_narrative_system_prompt() -> str:
    """Return the system prompt for Phase 1 narrative generation.

    Instructs the LLM to transform structured customer features into a
    concise, structured narrative that highlights satisfaction signals.

    Returns:
        System prompt string.
    """
    return (
        "You are a customer experience analyst. "
        "You receive structured e-commerce data for a single customer and "
        "produce a clear, concise narrative summary. "
        "Always respond in the following exact format — do not add extra sections:\n\n"
        "Customer Summary:\n"
        "<2-3 sentence overview of the customer's purchase history>\n\n"
        "Key Positive Signals:\n"
        "* <signal>\n"
        "* <signal>\n\n"
        "Key Negative Signals:\n"
        "* <signal>\n"
        "* <signal>\n\n"
        "Notable Events:\n"
        "* <specific event, e.g. an order delivered 20 days late>\n\n"
        "If a section has nothing to report, write '* None'."
    )


def build_narrative_user_prompt(features: dict, customer_id: Optional[str] = None) -> str:
    """Build the Phase 1 user prompt from structured customer features.

    Args:
        features: Dict of aggregated customer-level metrics.
        customer_id: Optional identifier included as a comment for tracing.

    Returns:
        User prompt string.
    """
    cid_line = f"Customer ID: {customer_id}\n" if customer_id else ""
    return (
        f"{cid_line}"
        f"Number of orders: {features.get('num_orders', 'N/A')}\n"
        f"Average item price (BRL): {features.get('avg_price', 'N/A')}\n"
        f"Average freight value (BRL): {features.get('avg_freight', 'N/A')}\n"
        f"Average delivery time (days): {features.get('avg_delivery_time', 'N/A')}\n"
        f"Late delivery ratio: {features.get('late_delivery_ratio', 'N/A')}\n"
        f"Distinct product categories: {features.get('num_product_categories', 'N/A')}\n"
        f"Distinct payment types used: {features.get('num_payment_types', 'N/A')}\n"
        f"Max single delivery delay (days): {features.get('max_delay_days', 'N/A')}\n"
        f"Total spend (BRL): {features.get('total_spend', 'N/A')}\n\n"
        "Please generate the structured customer narrative."
    )


# ---------------------------------------------------------------------------
# Phase 2 – Satisfaction prediction
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """Return the system prompt for Phase 2 satisfaction probability prediction.

    Instructs the LLM to reason about customer satisfaction and output a
    probability in the range [0, 1] where 1 = definitely satisfied
    (review score 4 or 5).

    Returns:
        System prompt string.
    """
    return (
        "You are an expert in e-commerce customer satisfaction prediction. "
        "You will receive a narrative summary of a customer's purchase history. "
        "Based on the narrative, estimate the probability that this customer's "
        "overall experience results in a POSITIVE review (score 4 or 5 out of 5).\n\n"
        "Respond in this exact format — nothing else:\n\n"
        "Reasoning:\n"
        "<2-4 sentences explaining the key factors driving your estimate>\n\n"
        "Probability:\n"
        "<a single decimal number between 0 and 1, e.g. 0.73>"
    )


def build_user_prompt(profile_text: str, customer_id: Optional[str] = None) -> str:
    """Build the Phase 2 user prompt from the Phase 1 narrative.

    Args:
        profile_text: Natural language narrative from Phase 1.
        customer_id: Optional identifier for debugging/tracing.

    Returns:
        User prompt string.
    """
    cid_line = f"[Customer ID: {customer_id}]\n" if customer_id else ""
    return (
        f"{cid_line}"
        "Based on the following customer narrative, estimate the probability "
        "of a positive review (score 4 or 5):\n\n"
        f"{profile_text}\n\n"
        "Respond with Reasoning and Probability as instructed."
    )


def build_prompt_pair(profile_text: str, customer_id: Optional[str] = None) -> tuple[str, str]:
    """Convenience wrapper: return (system_prompt, user_prompt) for Phase 2.

    Args:
        profile_text: Customer narrative from Phase 1.
        customer_id: Optional customer identifier.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    return build_system_prompt(), build_user_prompt(profile_text, customer_id)
