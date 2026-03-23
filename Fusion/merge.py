from __future__ import annotations

from pathlib import Path

import pandas as pd


# Default relative paths (from this file's folder)
DATA_DIR = Path(__file__).resolve().parent / "data"
FE_FILE = DATA_DIR / "fe.csv"
LLM_FILE = DATA_DIR / "llm.csv"
STOCK_FILE = DATA_DIR / "data_stock.csv"

# Soft voting weights (must sum to 1.0)
WEIGHT_FE = 0.5
WEIGHT_LLM = 0.5


def load_and_prepare_data(
	fe_path: Path = FE_FILE,
	llm_path: Path = LLM_FILE,
	stock_path: Path = STOCK_FILE,
) -> pd.DataFrame:
	"""Load FE, LLM, and ground-truth files; return merged frame ready for evaluation."""
	fe = pd.read_csv(fe_path)
	llm = pd.read_csv(llm_path)
	stock = pd.read_csv(stock_path)

	fe = fe.rename(
		columns={
			"customer_unique_id": "customer_id",
			"predicted_satisfaction": "p_fe",
		}
	)
	llm = llm.rename(columns={"P_LLM": "p_llm"})
	stock = stock.rename(columns={"customer_unique_id": "customer_id"})

	# Keep only required columns and force numeric probability/score types.
	fe = fe[["customer_id", "p_fe"]].copy()
	llm = llm[["customer_id", "p_llm"]].copy()
	stock = stock[["customer_id", "review_score"]].copy()

	fe["p_fe"] = pd.to_numeric(fe["p_fe"], errors="coerce")
	llm["p_llm"] = pd.to_numeric(llm["p_llm"], errors="coerce")
	stock["review_score"] = pd.to_numeric(stock["review_score"], errors="coerce")

	merged = (
		stock.merge(fe, on="customer_id", how="inner")
		.merge(llm, on="customer_id", how="inner")
		.dropna(subset=["review_score", "p_fe", "p_llm"])
	)

	return merged


def evaluate_soft_voting(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
	"""Apply soft voting and return accuracy + prediction dataframe."""
	if not 0 <= WEIGHT_FE <= 1 or not 0 <= WEIGHT_LLM <= 1:
		raise ValueError("WEIGHT_FE and WEIGHT_LLM must be in [0, 1].")

	if abs((WEIGHT_FE + WEIGHT_LLM) - 1.0) > 1e-9:
		raise ValueError("WEIGHT_FE + WEIGHT_LLM must equal 1.0.")

	out = df.copy()

	# Ground truth rule: review_score 1-3 => 0 (not satisfied), 4-5 => 1 (satisfied).
	out["y_true"] = (out["review_score"] >= 4).astype(int)

	# Soft voting probability and final class with threshold 0.5.
	out["p_soft"] = WEIGHT_FE * out["p_fe"] + WEIGHT_LLM * out["p_llm"]
	out["y_pred"] = (out["p_soft"] >= 0.5).astype(int)

	acc = (out["y_true"] == out["y_pred"]).mean()
	return acc, out


def main() -> None:
	merged = load_and_prepare_data()
	accuracy, result = evaluate_soft_voting(merged)

	print(f"Rows evaluated: {len(result)}")
	print(f"Soft Voting Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


if __name__ == "__main__":
	main()
