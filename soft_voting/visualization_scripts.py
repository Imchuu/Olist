from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
METRICS_JSON = DATA_DIR / "evaluation_metrics.json"


def load_metrics(path: Path = METRICS_JSON) -> tuple[pd.DataFrame, dict]:
	"""Load evaluation json and return metric frame + raw payload."""
	with path.open("r", encoding="utf-8") as f:
		payload = json.load(f)

	models = payload.get("models", {})
	frame = pd.DataFrame.from_dict(models, orient="index")
	return frame, payload


def ensure_results_dir(path: Path = RESULTS_DIR) -> None:
	path.mkdir(parents=True, exist_ok=True)


def plot_confusion_from_metrics(df: pd.DataFrame, output_path: Path) -> None:
	"""Plot 2x2 confusion heatmaps for all models based on TP/TN/FP/FN counts."""
	model_names = df.index.tolist()
	fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))

	if len(model_names) == 1:
		axes = [axes]

	for ax, model in zip(axes, model_names):
		tp = int(df.loc[model, "tp"])
		tn = int(df.loc[model, "tn"])
		fp = int(df.loc[model, "fp"])
		fn = int(df.loc[model, "fn"])

		matrix = [[tp, fp], [fn, tn]]
		im = ax.imshow(matrix, cmap="Blues")
		ax.set_title(model)
		ax.set_xticks([0, 1])
		ax.set_yticks([0, 1])
		ax.set_xticklabels(["Thuc: Hai long", "Thuc: Khong hai long"])
		ax.set_yticklabels(["Du doan: Hai long", "Du doan: Khong hai long"])

		for i in range(2):
			for j in range(2):
				# Use white text on dark cells for better contrast.
				text_color = "white" if matrix[i][j] >= (max(map(max, matrix)) * 0.6) else "black"
				ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color=text_color, fontweight="bold")

		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

	fig.suptitle("Confusion Matrix by Model", fontsize=14)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def plot_metric_comparison(df: pd.DataFrame, output_path: Path) -> None:
	"""Plot one grouped chart with Accuracy, F1 Score, AUC, Precision in a single figure."""
	metrics = ["accuracy", "f1_score", "auc", "precision"]
	metric_labels = ["Accuracy", "F1 Score", "AUC", "Precision"]
	model_names = df.index.tolist()

	x = np.arange(len(model_names))
	bar_width = 0.18
	# Palette aligned with the sample style: blue, teal, purple, orange.
	colors = ["#5B9BD5", "#41B39A", "#B5659A", "#F2A93B"]

	fig, ax = plt.subplots(figsize=(12, 7))
	for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
		values = [float(df.loc[model, metric]) for model in model_names]
		offset = (i - (len(metrics) - 1) / 2) * bar_width
		bars = ax.bar(x + offset, values, width=bar_width, label=label, color=colors[i])

		for bar in bars:
			height = bar.get_height()
			ax.annotate(
				f"{height:.4f}",
				(bar.get_x() + bar.get_width() / 2, height),
				ha="center",
				va="bottom",
				fontsize=8,
			)

	ax.set_title("Detailed Metrics Comparison")
	ax.set_ylabel("Score")
	ax.set_xlabel("Method")
	ax.set_xticks(x)
	ax.set_xticklabels(model_names, rotation=0)
	ax.set_ylim(0, 1.05)
	ax.grid(axis="y", alpha=0.3)
	ax.legend(loc="lower left", title="Metric")

	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def plot_rmse(df: pd.DataFrame, output_path: Path) -> None:
	"""Plot RMSE where lower is better."""
	rmse_df = df[["rmse"]].copy()
	ax = rmse_df.plot(kind="bar", figsize=(8, 5), legend=False, color="#e67e22")
	ax.set_title("RMSE by Model (Lower is Better)")
	ax.set_ylabel("RMSE")
	ax.set_xlabel("Model")
	ax.grid(axis="y", alpha=0.3)

	for patch in ax.patches:
		height = patch.get_height()
		ax.annotate(f"{height:.4f}", (patch.get_x() + patch.get_width() / 2, height),
					ha="center", va="bottom", fontsize=9)

	fig = ax.get_figure()
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def plot_outcome_breakdown(df: pd.DataFrame, output_path: Path) -> None:
	"""Plot stacked bar for TP/TN/FP/FN counts by model."""
	outcome_df = df[["tp", "tn", "fp", "fn"]].copy()
	ax = outcome_df.plot(kind="bar", stacked=True, figsize=(12, 6))
	ax.set_title("Prediction Outcome Breakdown (TP/TN/FP/FN)")
	ax.set_ylabel("Count")
	ax.set_xlabel("Model")
	ax.grid(axis="y", alpha=0.3)
	ax.legend(loc="upper right")

	fig = ax.get_figure()
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def main() -> None:
	ensure_results_dir()
	df, payload = load_metrics()

	rows = payload.get("rows_evaluated", "unknown")
	print(f"Rows evaluated: {rows}")

	confusion_path = RESULTS_DIR / "confusion_matrices_from_json.png"
	metrics_path = RESULTS_DIR / "core_metrics_comparison_overview.png"
	rmse_path = RESULTS_DIR / "rmse_comparison.png"
	outcome_path = RESULTS_DIR / "outcome_breakdown_stacked.png"

	plot_confusion_from_metrics(df, confusion_path)
	plot_metric_comparison(df, metrics_path)
	plot_rmse(df, rmse_path)
	plot_outcome_breakdown(df, outcome_path)

	print("Generated PNG files:")
	print(f"- {confusion_path}")
	print(f"- {metrics_path}")
	print(f"- {rmse_path}")
	print(f"- {outcome_path}")


if __name__ == "__main__":
	main()
