"""Load order-level dataset for Flow 2 LLM pipeline."""

import logging
from pathlib import Path
from typing import Union

import pandas as pd

from .config import DATETIME_COLUMNS, REQUIRED_DATASET_COLUMNS

logger = logging.getLogger(__name__)

def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """Load the order-level dataset from disk.

    Expects a CSV with order-level rows and columns relevant to customer
    satisfaction. Handles encoding and coerces datetime columns.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with order-level data, datetime columns parsed.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)

    for col in DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """Validate that the loaded dataset has the expected structure.

    Args:
        df: The loaded DataFrame.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If required columns are missing or data quality fails.
    """
    missing = [c for c in REQUIRED_DATASET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    if df["customer_unique_id"].isnull().all():
        raise ValueError("Column 'customer_unique_id' is entirely null.")

    if df["review_score"].isnull().mean() > 0.5:
        raise ValueError("More than 50% of review_score values are missing.")

    logger.info("Dataset validation passed.")
    return True
