"""Utility functions for Flow 2 LLM pipeline."""

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_output_dir(path: Union[str, Path]) -> Path:
    """Ensure the parent directory of the given path exists.

    Creates the directory structure if it does not exist.

    Args:
        path: File path whose parent directory will be created.

    Returns:
        Resolved Path object for the given path.

    Raises:
        PermissionError: If the directory cannot be created due to permissions.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured output directory: %s", path.parent)
    return path


def save_predictions(
    customer_ids: list[str],
    probabilities: list[float],
    output_path: Union[str, Path],
) -> None:
    """Save customer IDs and LLM probabilities to a CSV file.

    Output format: columns 'customer_id' and 'P_LLM'.
    Designed for easy merging with XGBoost predictions on 'customer_id'.

    Args:
        customer_ids: List of customer identifiers.
        probabilities: List of P(satisfied) values in [0, 1].
        output_path: Destination path for the CSV file.
    """
    if len(customer_ids) != len(probabilities):
        raise ValueError(
            f"Length mismatch: {len(customer_ids)} IDs vs {len(probabilities)} probabilities"
        )

    output_path = ensure_output_dir(output_path)
    df = pd.DataFrame({"customer_id": customer_ids, "P_LLM": probabilities})
    df.to_csv(output_path, index=False)
    logger.info("Saved %d predictions to %s", len(df), output_path)


def load_predictions(path: Union[str, Path]) -> pd.DataFrame:
    """Load previously saved LLM predictions from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with columns 'customer_id' and 'P_LLM'.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If expected columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    df = pd.read_csv(path)
    for col in ("customer_id", "P_LLM"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in predictions file.")

    logger.info("Loaded %d predictions from %s", len(df), path)
    return df


def merge_with_xgb(
    llm_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
    llm_weight: float = 0.5,
    xgb_weight: float = 0.5,
) -> pd.DataFrame:
    """Merge LLM and XGBoost predictions using soft voting.

    Both DataFrames must have a 'customer_id' column.
    LLM DataFrame must have 'P_LLM'; XGBoost DataFrame must have 'P_XGB'.

    Args:
        llm_df: DataFrame with columns 'customer_id' and 'P_LLM'.
        xgb_df: DataFrame with columns 'customer_id' and 'P_XGB'.
        llm_weight: Weight for LLM probability (default 0.5).
        xgb_weight: Weight for XGBoost probability (default 0.5).

    Returns:
        Merged DataFrame with columns 'customer_id', 'P_LLM', 'P_XGB',
        and 'P_ensemble' (weighted average).
    """
    merged = pd.merge(llm_df, xgb_df, on="customer_id", how="inner")
    merged["P_ensemble"] = (
        llm_weight * merged["P_LLM"] + xgb_weight * merged["P_XGB"]
    )
    logger.info(
        "Merged %d LLM + %d XGB → %d common customers.",
        len(llm_df), len(xgb_df), len(merged),
    )
    return merged
