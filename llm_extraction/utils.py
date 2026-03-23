"""Utility functions for Flow 2 LLM pipeline."""

import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any, Union

import pandas as pd

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressBar:
    """Simple terminal progress bar with elapsed time, speed and ETA."""

    def __init__(self, total: int, label: str, width: int = 28) -> None:
        self.total = max(0, int(total))
        self.label = label
        self.width = max(10, int(width))
        self.start_time = time.perf_counter()
        self.last_rendered = ""

    def update(self, completed: int) -> None:
        """Render the current progress state in-place."""
        completed = max(0, min(int(completed), self.total if self.total > 0 else int(completed)))
        elapsed = time.perf_counter() - self.start_time

        if self.total <= 0:
            line = f"{self.label} | {'#' * self.width} | 100.0% 0/0 elapsed 00:00:00 eta 00:00:00"
        else:
            ratio = completed / self.total
            filled = int(self.width * ratio)
            bar = "#" * filled + "-" * (self.width - filled)
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta_seconds = (self.total - completed) / rate if rate > 0 else 0.0
            line = (
                f"{self.label} | {bar} | {ratio * 100:6.2f}% "
                f"{completed}/{self.total} elapsed {_format_duration(elapsed)} "
                f"eta {_format_duration(eta_seconds)}"
            )

        # Pad with spaces to overwrite previous longer content when shrinking.
        padding = max(0, len(self.last_rendered) - len(line))
        sys.stdout.write("\r" + line + (" " * padding))
        sys.stdout.flush()
        self.last_rendered = line

    def finish(self) -> None:
        """Complete the bar and move cursor to the next line."""
        self.update(self.total)
        sys.stdout.write("\n")
        sys.stdout.flush()


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


def append_csv_row(
    output_path: Union[str, Path],
    fieldnames: list[str],
    row: dict[str, Any],
) -> None:
    """Append a single row to CSV and create file/header on first write."""
    path = ensure_output_dir(output_path)
    needs_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def load_existing_value_map(
    path: Union[str, Path],
    id_column: str,
    value_column: str,
) -> dict[str, Any]:
    """Load an existing CSV checkpoint into an ID->value map."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}

    try:
        df = pd.read_csv(p)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read checkpoint %s: %s", p, exc)
        return {}

    if id_column not in df.columns or value_column not in df.columns:
        logger.warning(
            "Checkpoint %s missing columns '%s' or '%s'; ignoring existing file.",
            p,
            id_column,
            value_column,
        )
        return {}

    mapping: dict[str, Any] = {}
    for _, row in df[[id_column, value_column]].dropna(subset=[id_column]).iterrows():
        mapping[str(row[id_column])] = row[value_column]
    return mapping


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
