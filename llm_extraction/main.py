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

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ML.config import (  # type: ignore
        INPUT_PATH,
        LOCAL_API_BASE_URL,
        MAX_CUSTOMERS,
        MAX_WORKERS,
        LLM_MODEL_PHASE1,
        LLM_MODEL_PHASE2,
        LLM_TEMPERATURE,
        OUTPUT_PATH,
        PHASE2_MAX_TOKENS,
        PHASE1_OUTPUT_PATH,
        PROGRESS_LOG_EVERY_N,
        STOP_AFTER_PHASE1,
    )
    from ML.data_loader import load_dataset, validate_dataset  # type: ignore
    from ML.feature_aggregator import aggregate_to_customer_level  # type: ignore
    from ML.profile_generator import generate_all_profiles  # type: ignore
    from ML.prompt_builder import build_system_prompt, build_user_prompt  # type: ignore
    from ML.llm_predictor import call_llm, check_local_llm_connection  # type: ignore
    from ML.probability_parser import parse_probability_from_response  # type: ignore
    from ML.utils import (  # type: ignore
        ProgressBar,
        append_csv_row,
        ensure_output_dir,
        load_existing_value_map,
        save_predictions,
    )
else:
    from .config import (
        INPUT_PATH,
        LOCAL_API_BASE_URL,
        MAX_CUSTOMERS,
        MAX_WORKERS,
        LLM_MODEL_PHASE1,
        LLM_MODEL_PHASE2,
        LLM_TEMPERATURE,
        OUTPUT_PATH,
        PHASE2_MAX_TOKENS,
        PHASE1_OUTPUT_PATH,
        PROGRESS_LOG_EVERY_N,
        STOP_AFTER_PHASE1,
    )
    from .data_loader import load_dataset, validate_dataset
    from .feature_aggregator import aggregate_to_customer_level
    from .profile_generator import generate_all_profiles
    from .prompt_builder import build_system_prompt, build_user_prompt
    from .llm_predictor import call_llm, check_local_llm_connection
    from .probability_parser import parse_probability_from_response
    from .utils import (
        ProgressBar,
        append_csv_row,
        ensure_output_dir,
        load_existing_value_map,
        save_predictions,
    )

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
    input_path = INPUT_PATH
    logger.info("=== Flow 2 — Phase 1: Narrative Generation ===")
    logger.info("Loading dataset: %s", input_path)
    orders_df = load_dataset(input_path)
    validate_dataset(orders_df)

    # ------------------------------------------------------------------
    # Phase 1 – Step 2: Aggregate to customer level
    # ------------------------------------------------------------------
    customers_df = aggregate_to_customer_level(orders_df)
    if MAX_CUSTOMERS > 0:
        customers_df = customers_df.head(MAX_CUSTOMERS).copy()
        logger.info("Benchmark mode: limiting to first %d customers", len(customers_df))
    logger.info("Customer-level features ready: %d customers", len(customers_df))

    # Phase 1 should validate the model used by narrative generation.
    preflight_model = LLM_MODEL_PHASE1
    if not check_local_llm_connection(required_model=preflight_model):
        raise ConnectionError(
            f"Cannot connect to LM Studio at {LOCAL_API_BASE_URL}. "
            f"Please start LM Studio local server and load model '{LLM_MODEL_PHASE1}'."
        )

    # ------------------------------------------------------------------
    # Phase 1 – Step 3: Generate narratives (LLM #1)
    # ------------------------------------------------------------------
    phase1_output = PHASE1_OUTPUT_PATH
    existing_profiles = load_existing_value_map(
        path=phase1_output,
        id_column="customer_id",
        value_column="profile_text",
    )
    if existing_profiles:
        logger.info("Found %d existing Phase 1 narratives. Will resume from checkpoint.", len(existing_profiles))

    def _on_phase1_generated(customer_id: str, profile_text: str) -> None:
        append_csv_row(
            output_path=phase1_output,
            fieldnames=["customer_id", "profile_text"],
            row={"customer_id": customer_id, "profile_text": profile_text},
        )

    customers_with_profiles = generate_all_profiles(
        customers_df,
        existing_profiles=existing_profiles,
        on_profile_generated=_on_phase1_generated,
    )

    if STOP_AFTER_PHASE1:
        ensure_output_dir(phase1_output)
        phase1_df = customers_with_profiles[["customer_id", "profile_text"]].copy()
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
    logger.info("Phase 2 using %d worker(s)", MAX_WORKERS)

    # Before Phase 2 starts, validate that the Phase 2 model is available.
    if not check_local_llm_connection(required_model=LLM_MODEL_PHASE2):
        raise ConnectionError(
            f"Phase 1 completed. Cannot start Phase 2 because model '{LLM_MODEL_PHASE2}' "
            f"is not loaded in LM Studio at {LOCAL_API_BASE_URL}. "
            "Please load the Phase 2 model, then rerun."
        )

    system_prompt = build_system_prompt()
    model = LLM_MODEL_PHASE2
    temperature = LLM_TEMPERATURE

    rows = customers_with_profiles[["customer_id", "profile_text"]].to_dict("records")
    total = len(rows)
    customer_ids: list[str] = [""] * total
    probabilities: list[float] = [0.5] * total
    progress_every = PROGRESS_LOG_EVERY_N
    progress_bar = ProgressBar(total=total, label="Phase 2")

    output_path = OUTPUT_PATH
    existing_phase2 = load_existing_value_map(
        path=output_path,
        id_column="customer_id",
        value_column="P_LLM",
    )
    if existing_phase2:
        logger.info("Found %d existing Phase 2 predictions. Will resume from checkpoint.", len(existing_phase2))

    def _predict_one(idx: int, row_dict: dict) -> tuple[int, str, float]:
        cid = str(row_dict["customer_id"])
        narrative = str(row_dict["profile_text"])
        user_prompt = build_user_prompt(profile_text=narrative, customer_id=cid)

        try:
            response_text = call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=PHASE2_MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Phase 2 LLM call failed for customer %s: %s", cid, exc)
            response_text = ""

        prob = parse_probability_from_response(response_text)

        # If parsing fails, run a lightweight second pass that only extracts
        # one probability number from the same response text.
        if prob is None and response_text.strip():
            extraction_system_prompt = (
                "You are a strict numeric extractor. Return only one probability number "
                "between 0 and 1 with 6 decimal places. No extra text."
            )
            extraction_user_prompt = (
                "Extract the final probability from this model response. "
                "If no valid probability exists, return exactly 0.500000.\n\n"
                f"{response_text}"
            )
            try:
                extraction_text = call_llm(
                    system_prompt=extraction_system_prompt,
                    user_prompt=extraction_user_prompt,
                    model=model,
                    temperature=0.0,
                    max_tokens=24,
                )
                prob = parse_probability_from_response(extraction_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Probability extraction retry failed for customer %s: %s", cid, exc)

        if prob is None:
            prob = 0.5
            logger.warning("Using default probability 0.50 for customer %s after parse retries.", cid)

        return idx, cid, prob

    completed = 0
    if MAX_WORKERS <= 1:
        for i, row_dict in enumerate(rows):
            cid = str(row_dict["customer_id"])
            if cid in existing_phase2:
                customer_ids[i] = cid
                probabilities[i] = float(existing_phase2[cid])
                completed += 1
                progress_bar.update(completed)
                if completed % progress_every == 0 or completed == total:
                    logger.info("  Phase 2 progress: %d / %d", completed, total)
                continue

            idx, cid, prob = _predict_one(i, row_dict)
            customer_ids[idx] = cid
            probabilities[idx] = prob
            append_csv_row(
                output_path=output_path,
                fieldnames=["customer_id", "P_LLM"],
                row={"customer_id": cid, "P_LLM": prob},
            )
            completed += 1
            progress_bar.update(completed)
            if completed % progress_every == 0 or completed == total:
                logger.info("  Phase 2 progress: %d / %d", completed, total)
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i, row_dict in enumerate(rows):
                cid = str(row_dict["customer_id"])
                if cid in existing_phase2:
                    customer_ids[i] = cid
                    probabilities[i] = float(existing_phase2[cid])
                    completed += 1
                    progress_bar.update(completed)
                    if completed % progress_every == 0 or completed == total:
                        logger.info("  Phase 2 progress: %d / %d", completed, total)
                    continue
                futures.append(executor.submit(_predict_one, i, row_dict))

            for future in as_completed(futures):
                idx, cid, prob = future.result()
                customer_ids[idx] = cid
                probabilities[idx] = prob
                append_csv_row(
                    output_path=output_path,
                    fieldnames=["customer_id", "P_LLM"],
                    row={"customer_id": cid, "P_LLM": prob},
                )
                completed += 1
                progress_bar.update(completed)
                if completed % progress_every == 0 or completed == total:
                    logger.info("  Phase 2 progress: %d / %d", completed, total)

    progress_bar.finish()

    # ------------------------------------------------------------------
    # Step 6: Save results
    # ------------------------------------------------------------------
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
