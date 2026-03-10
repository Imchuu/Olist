"""Build LLM prompts for Phase 1 (narrative) and Phase 2 (prediction)."""

from typing import Optional


# ---------------------------------------------------------------------------
# Phase 1 – Narrative generation
# ---------------------------------------------------------------------------

def build_narrative_system_prompt() -> str:
    return (
        "You convert structured e-commerce customer metrics into a strict, "
        "fixed-format customer behavior summary.\n\n"

        "Rules:\n"
        "- Only describe the provided metrics.\n"
        "- Do NOT infer satisfaction probability.\n"
        "- Do NOT add assumptions.\n"
        "- Keep all numeric values exactly as provided.\n"
        "- Include every metric exactly once in the metric lines.\n"
        "- Add exactly one final natural-language sentence in the Summary line.\n"
        "- Do not add any extra fields or commentary.\n\n"

        "Output format:\n\n"
        "Customer Behavior Summary\n\n"
        "Orders: <number>\n"
        "Total spend: <value> BRL\n"
        "Average item price: <value> BRL\n"
        "Average freight value: <value> BRL\n"
        "Average delivery time: <value> days\n"
        "Late delivery ratio: <value>\n"
        "Maximum delivery delay: <value> days\n"
        "Product categories purchased: <number>\n"
        "Payment types used: <number>\n"
        "Summary: <one sentence summary in natural language>"
    )


def build_narrative_user_prompt(features: dict, customer_id: Optional[str] = None) -> str:
    cid_line = f"Customer ID: {customer_id}\n" if customer_id else ""
    return (
        f"{cid_line}"
        "Customer purchase metrics:\n\n"
        f"Orders: {features.get('num_orders', 'N/A')}\n"
        f"Total spend: {features.get('total_spend', 'N/A')} BRL\n"
        f"Average item price: {features.get('avg_price', 'N/A')} BRL\n"
        f"Average freight value: {features.get('avg_freight', 'N/A')} BRL\n"
        f"Average delivery time: {features.get('avg_delivery_time', 'N/A')} days\n"
        f"Late delivery ratio: {features.get('late_delivery_ratio', 'N/A')}\n"
        f"Maximum delivery delay: {features.get('max_delay_days', 'N/A')} days\n"
        f"Product categories purchased: {features.get('num_product_categories', 'N/A')}\n"
        f"Payment types used: {features.get('num_payment_types', 'N/A')}\n\n"
        "Generate output in the exact required format and include exactly one Summary sentence."
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
