"""Flow 2: Two-phase LLM customer satisfaction probability estimation.

Pipeline:
  Phase 1 – Narrative Generation
    1. Load order-level dataset
    2. Aggregate to customer-level features
    3. Call LLM #1 to convert features → structured natural language narrative

  Phase 2 – Satisfaction Prediction
    4. Call LLM #2 with the narrative to estimate P(positive review)
    5. Parse probability from response
    6. Save results (customer_id, P_LLM) to results/llm_predictions.csv

The output can be merged with XGBoost predictions (customer_id, P_XGB)
via utils.merge_with_xgb() for soft voting ensemble.

Run from project root:
    python -m ML.main
"""

import logging
import sys

import pandas as pd

from .config import (
    get_input_path,
    get_output_path,
    get_phase1_output_path,
    get_stop_after_phase1,
    get_llm_model,
    get_llm_temperature,
)
from .data_loader import load_dataset, validate_dataset
from .feature_aggregator import aggregate_to_customer_level
from .profile_generator import generate_all_profiles
from .prompt_builder import build_system_prompt, build_user_prompt
from .llm_predictor import call_llm
from .probability_parser import safe_parse_probability
from .utils import ensure_output_dir, save_predictions

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run_flow2_pipeline() -> pd.DataFrame:
    """Execute the full Flow 2 pipeline from dataset load to saved predictions.

    Returns:
        DataFrame with columns 'customer_id' and 'P_LLM'.
    """
    # ------------------------------------------------------------------
    # Phase 1 – Step 1: Load dataset
    # ------------------------------------------------------------------
    input_path = get_input_path()
    logger.info("=== Flow 2 — Phase 1: Narrative Generation ===")
    logger.info("Loading dataset: %s", input_path)
    orders_df = load_dataset(input_path)
    validate_dataset(orders_df)

    # ------------------------------------------------------------------
    # Phase 1 – Step 2: Aggregate to customer level
    # ------------------------------------------------------------------
    customers_df = aggregate_to_customer_level(orders_df)
    logger.info("Customer-level features ready: %d customers", len(customers_df))

    # ------------------------------------------------------------------
    # Phase 1 – Step 3: Generate narratives (LLM #1)
    # ------------------------------------------------------------------
    customers_with_profiles = generate_all_profiles(customers_df)

    if get_stop_after_phase1():
        phase1_output = get_phase1_output_path()
        ensure_output_dir(phase1_output)
        phase1_df = customers_with_profiles[["customer_id", "profile_text"]].copy()
        phase1_df.to_csv(phase1_output, index=False)
        logger.info(
            "Stop-after-Phase-1 mode is ON. Saved %d narratives to %s",
            len(phase1_df),
            phase1_output,
        )
        return phase1_df

    # ------------------------------------------------------------------
    # Phase 2 – Steps 4-5: Predict satisfaction probability
    # ------------------------------------------------------------------
    logger.info("=== Flow 2 — Phase 2: Satisfaction Prediction ===")
    system_prompt = build_system_prompt()
    model = get_llm_model()
    temperature = get_llm_temperature()

    customer_ids: list[str] = []
    probabilities: list[float] = []
    total = len(customers_with_profiles)

    for _, row in customers_with_profiles.iterrows():
        cid = str(row["customer_id"])
        narrative = str(row["profile_text"])

        user_prompt = build_user_prompt(profile_text=narrative, customer_id=cid)

        try:
            response_text = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Phase 2 LLM call failed for customer %s: %s", cid, exc)
            response_text = ""

        prob = safe_parse_probability(response_text, default=0.5)

        customer_ids.append(cid)
        probabilities.append(prob)

        if (len(customer_ids)) % 50 == 0 or len(customer_ids) == total:
            logger.info("  Phase 2 progress: %d / %d", len(customer_ids), total)

    # ------------------------------------------------------------------
    # Step 6: Save results
    # ------------------------------------------------------------------
    output_path = get_output_path()
    ensure_output_dir(output_path)
    save_predictions(
        customer_ids=customer_ids,
        probabilities=probabilities,
        output_path=output_path,
    )

    results_df = pd.DataFrame({"customer_id": customer_ids, "P_LLM": probabilities})
    logger.info("=== Flow 2 complete. %d predictions saved to %s ===", len(results_df), output_path)
    return results_df


def main() -> None:
    """Entry point: run the Flow 2 pipeline."""
    results_df = run_flow2_pipeline()
    if "P_LLM" in results_df.columns:
        print(f"\nFlow 2 complete — {len(results_df)} predictions saved.")
    else:
        print(f"\nPhase 1 complete — {len(results_df)} narratives saved.")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
