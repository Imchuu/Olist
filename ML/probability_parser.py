"""Parse the probability value from Phase 2 LLM response text."""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Matches "Probability:\n0.73" or "Probability: 0.73" with optional whitespace
_PROB_PATTERN = re.compile(
    r"probability\s*[:\-]\s*([0-1](?:\.\d+)?)",
    re.IGNORECASE,
)

# Fallback: any standalone decimal in [0, 1] at end of response
_FALLBACK_PATTERN = re.compile(
    r"\b(0\.\d{1,4}|1\.0{1,4})\b",
    re.IGNORECASE,
)


def parse_probability_from_response(text: str) -> Optional[float]:
    """Extract the probability value from a Phase 2 LLM response.

    Expected format:
        Reasoning:
        <text>

        Probability:
        <float between 0 and 1>

    Falls back to scanning the entire response for any decimal in [0, 1]
    if the primary pattern does not match.

    Args:
        text: Raw LLM response string.

    Returns:
        Float probability in [0, 1], or None if no valid number found.
    """
    if not text or not text.strip():
        logger.warning("parse_probability_from_response received empty text.")
        return None

    # Primary: look for "Probability: <value>"
    match = _PROB_PATTERN.search(text)
    if match:
        value = float(match.group(1))
        value = max(0.0, min(1.0, value))
        logger.debug("Parsed probability (primary): %.4f", value)
        return value

    # Fallback: grab the last decimal in [0, 1] found in the response
    all_matches = _FALLBACK_PATTERN.findall(text)
    if all_matches:
        value = float(all_matches[-1])
        value = max(0.0, min(1.0, value))
        logger.warning(
            "Probability not found via primary pattern; fallback value: %.4f", value
        )
        return value

    logger.error("Could not parse any probability from response:\n%s", text[:300])
    return None


def safe_parse_probability(text: str, default: float = 0.5) -> float:
    """Parse probability with a guaranteed non-None return value.

    Args:
        text: Raw LLM response string.
        default: Value returned when parsing completely fails (default 0.5).

    Returns:
        Parsed probability or the default fallback.
    """
    prob = parse_probability_from_response(text)
    if prob is None:
        logger.warning("Using default probability %.2f due to parse failure.", default)
        return default
    return prob
