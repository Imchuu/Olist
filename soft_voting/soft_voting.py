from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd


# Default relative paths (from this file's folder)
DATA_DIR = Path(__file__).resolve().parent / "data"
FE_FILE = DATA_DIR / "fe.csv"
LLM_FILE = DATA_DIR / "llm.csv"
STOCK_FILE = DATA_DIR / "data_stock.csv"
CONFUSION_FIG_FILE = DATA_DIR / "visualization_nx_confusion.png"
METRICS_JSON_FILE = DATA_DIR / "evaluation_metrics.json"

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


def evaluate_single_models(df: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
	"""Evaluate FE-only and LLM-only using the same binary threshold and labels."""
	out = df.copy()
	out["y_true"] = (out["review_score"] >= 4).astype(int)
	out["y_pred_fe"] = (out["p_fe"] >= 0.5).astype(int)
	out["y_pred_llm"] = (out["p_llm"] >= 0.5).astype(int)

	acc_fe = (out["y_true"] == out["y_pred_fe"]).mean()
	acc_llm = (out["y_true"] == out["y_pred_llm"]).mean()
	return acc_fe, acc_llm, out


def compute_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
	"""Compute ROC-AUC without external ML dependency using rank statistic."""
	y_true_int = y_true.astype(int)
	y_prob_num = pd.to_numeric(y_prob, errors="coerce")
	valid_mask = y_prob_num.notna()

	y_true_valid = y_true_int[valid_mask]
	y_prob_valid = y_prob_num[valid_mask]

	n_pos = int((y_true_valid == 1).sum())
	n_neg = int((y_true_valid == 0).sum())

	if n_pos == 0 or n_neg == 0:
		return float("nan")

	ranks = y_prob_valid.rank(method="average")
	rank_sum_pos = ranks[y_true_valid == 1].sum()
	auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2)) / (n_pos * n_neg)
	return float(auc)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict[str, float | int]:
	"""Return a robust set of binary-classification metrics."""
	y_true_int = y_true.astype(int)
	y_pred_int = y_pred.astype(int)
	y_prob_num = pd.to_numeric(y_prob, errors="coerce")

	tp = int(((y_true_int == 1) & (y_pred_int == 1)).sum())
	tn = int(((y_true_int == 0) & (y_pred_int == 0)).sum())
	fp = int(((y_true_int == 0) & (y_pred_int == 1)).sum())
	fn = int(((y_true_int == 1) & (y_pred_int == 0)).sum())

	total = tp + tn + fp + fn
	accuracy = (tp + tn) / total if total else float("nan")
	precision = tp / (tp + fp) if (tp + fp) else 0.0
	recall = tp / (tp + fn) if (tp + fn) else 0.0
	f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
	specificity = tn / (tn + fp) if (tn + fp) else 0.0
	balanced_accuracy = (recall + specificity) / 2
	rmse = float((((y_true_int - y_prob_num) ** 2).mean()) ** 0.5)
	auc = compute_auc(y_true_int, y_prob_num)

	return {
		"accuracy": float(accuracy),
		"auc": float(auc),
		"f1_score": float(f1),
		"rmse": float(rmse),
		"precision": float(precision),
		"recall": float(recall),
		"specificity": float(specificity),
		"balanced_accuracy": float(balanced_accuracy),
		"tp": tp,
		"tn": tn,
		"fp": fp,
		"fn": fn,
	}


def write_json_report(
	rows_evaluated: int,
	fe_metrics: dict[str, float | int],
	llm_metrics: dict[str, float | int],
	soft_metrics: dict[str, float | int],
	output_file: Path = METRICS_JSON_FILE,
) -> None:
	"""Write all model metrics to JSON file."""
	payload = {
		"generated_at_utc": datetime.now(timezone.utc).isoformat(),
		"rows_evaluated": int(rows_evaluated),
		"threshold": 0.5,
		"models": {
			"Only Features Engineering": fe_metrics,
			"Only LLM": llm_metrics,
			"Soft Voting": soft_metrics,
		},
	}

	with output_file.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, ensure_ascii=False)


def build_confusion_table(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
	"""Create 2x2 table with required business labels."""
	tP = int(((y_true == 1) & (y_pred == 1)).sum())
	tN = int(((y_true == 0) & (y_pred == 0)).sum())
	fP = int(((y_true == 0) & (y_pred == 1)).sum())
	fN = int(((y_true == 1) & (y_pred == 0)).sum())

	return pd.DataFrame(
		[
			[tP, fP],
			[fN, tN],
		],
		index=["Du doan: Hai long", "Du doan: Khong hai long"],
		columns=["Thuc: Hai long", "Thuc: Khong hai long"],
	)


def plot_confusion_matrices(
	y_true: pd.Series,
	y_pred_fe: pd.Series,
	y_pred_llm: pd.Series,
	y_pred_soft: pd.Series,
	output_file: Path = CONFUSION_FIG_FILE,
) -> None:
	"""Save visualization image of 2x2 confusion matrices for three models."""
	tables = {
		"Only FE": build_confusion_table(y_true, y_pred_fe),
		"Only LLM": build_confusion_table(y_true, y_pred_llm),
		"Soft Voting": build_confusion_table(y_true, y_pred_soft),
	}

	fig, axes = plt.subplots(1, 3, figsize=(18, 5))

	for ax, (title, table) in zip(axes, tables.items()):
		matrix = table.values
		im = ax.imshow(matrix, cmap="Blues")
		ax.set_title(title)
		ax.set_xticks([0, 1])
		ax.set_yticks([0, 1])
		ax.set_xticklabels(table.columns)
		ax.set_yticklabels(table.index)

		for i in range(2):
			for j in range(2):
				ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

	fig.suptitle("Visualization nx - Bang 2x2 cho FE, LLM, Soft Voting", fontsize=14)
	fig.tight_layout()
	fig.savefig(output_file, dpi=180)
	plt.close(fig)


def main() -> None:
	merged = load_and_prepare_data()
	acc_fe, acc_llm, single_df = evaluate_single_models(merged)
	accuracy, result = evaluate_soft_voting(merged)

	fe_metrics = compute_metrics(single_df["y_true"], single_df["y_pred_fe"], single_df["p_fe"])
	llm_metrics = compute_metrics(single_df["y_true"], single_df["y_pred_llm"], single_df["p_llm"])
	soft_metrics = compute_metrics(result["y_true"], result["y_pred"], result["p_soft"])

	write_json_report(len(result), fe_metrics, llm_metrics, soft_metrics)

	fe_table = build_confusion_table(single_df["y_true"], single_df["y_pred_fe"])
	llm_table = build_confusion_table(single_df["y_true"], single_df["y_pred_llm"])
	soft_table = build_confusion_table(result["y_true"], result["y_pred"])

	plot_confusion_matrices(
		single_df["y_true"],
		single_df["y_pred_fe"],
		single_df["y_pred_llm"],
		result["y_pred"],
	)

	print(f"Rows evaluated: {len(result)}")
	print(f"FE Accuracy: {acc_fe:.4f} ({acc_fe * 100:.2f}%)")
	print(f"LLM Accuracy: {acc_llm:.4f} ({acc_llm * 100:.2f}%)")
	print(f"Soft Voting Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
	print("\nBang 2x2 - ONLY FE")
	print(fe_table.to_string())
	print("\nBang 2x2 - ONLY LLM")
	print(llm_table.to_string())
	print("\nBang 2x2 - SOFT VOTING")
	print(soft_table.to_string())
	print(f"\nVisualization saved to: {CONFUSION_FIG_FILE}")
	print(f"Metrics JSON saved to: {METRICS_JSON_FILE}")


if __name__ == "__main__":
	main()
