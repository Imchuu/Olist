# Hybrid ML–LLM Customer Satisfaction Prediction System

This repository implements an end-to-end customer satisfaction prediction system for the Olist use case, combining three complementary approaches:

1. Features Engineering + XGBoost
2. Two-phase LLM pipeline (narrative -> probability)
3. Soft Voting ensemble for model fusion and evaluation

Binary classification target:

- 0: Not satisfied (review score 1-3)
- 1: Satisfied (review score 4-5)

## Architecture Overview

High-level flow:

1. Build quantitative customer-level signals from transactional data (Features Engineering)
2. Generate natural-language customer profiles and estimate satisfaction probability with LLM
3. Fuse FE and LLM probabilities via Soft Voting
4. Evaluate and visualize both individual models and ensemble output

Soft voting formula:

$$
P_{soft} = w_{FE} \cdot P_{FE} + w_{LLM} \cdot P_{LLM}
$$

Current setup: $w_{FE}=0.5$, $w_{LLM}=0.5$.

## Repository Structure

```text
Olist/
|-- features_engineering/
|   |-- Olist_cleaned.ipynb
|   |-- EDA_Chuong3_1_final.ipynb
|   |-- Feature_engineering_clean.ipynb
|   |-- train file.ipynb
|   |-- validation/
|       |-- predict_satisfaction.py
|       |-- predict_acc.py
|
|-- llm_extraction/
|   |-- main.py
|   |-- config.py
|   |-- data_loader.py
|   |-- feature_aggregator.py
|   |-- prompt_builder.py
|   |-- llm_predictor.py
|   |-- probability_parser.py
|   |-- profile_generator.py
|   |-- utils.py
|   |-- processing.py
|   |-- data/
|   |-- results/
|
|-- soft_voting/
|   |-- soft_voting.py
|   |-- visualization_scripts.py
|   |-- data/
|   |-- results/
|
|-- data/
|-- requirement.txt
|-- README.md
```

## Component Details

### 1) Features Engineering

Primary notebooks:

- [features_engineering/Olist_cleaned.ipynb](features_engineering/Olist_cleaned.ipynb): base data cleaning and normalization
- [features_engineering/EDA_Chuong3_1_final.ipynb](features_engineering/EDA_Chuong3_1_final.ipynb): exploratory analysis and report figures
- [features_engineering/Feature_engineering_clean.ipynb](features_engineering/Feature_engineering_clean.ipynb): feature creation and refinement
- [features_engineering/train%20file.ipynb](features_engineering/train%20file.ipynb): model training experiments

Validation scripts:

- [features_engineering/validation/predict_satisfaction.py](features_engineering/validation/predict_satisfaction.py): train XGBoost, run cross-validation, export probabilities
- [features_engineering/validation/predict_acc.py](features_engineering/validation/predict_acc.py): compute Accuracy/AUC/F1 and generate ROC/PR/confusion plots

Typical output:

- `predicted_satisfaction_scores.csv` with columns `customer_unique_id`, `predicted_satisfaction`

### 2) LLM Extraction

Two-phase pipeline:

1. Phase 1: aggregate customer features and generate narratives
2. Phase 2: estimate satisfaction probability `P_LLM` from narratives

Core files:

- [llm_extraction/main.py](llm_extraction/main.py): pipeline entrypoint
- [llm_extraction/config.py](llm_extraction/config.py): model, path, worker, and timeout constants
- [llm_extraction/feature_aggregator.py](llm_extraction/feature_aggregator.py): order-level -> customer-level transformation
- [llm_extraction/prompt_builder.py](llm_extraction/prompt_builder.py): Phase 1 and Phase 2 prompt templates
- [llm_extraction/llm_predictor.py](llm_extraction/llm_predictor.py): LM Studio API client
- [llm_extraction/probability_parser.py](llm_extraction/probability_parser.py): robust probability extraction from model responses

Outputs:

- `llm_extraction/results/phase1_narratives.csv`
- `llm_extraction/results/llm_predictions.csv` (columns `customer_id`, `P_LLM`)

### 3) Soft Voting

Responsibilities:

- Merge FE and LLM predictions by customer identifier
- Evaluate three setups: Only FE, Only LLM, Soft Voting
- Export metrics JSON and visualization figures

Core files:

- [soft_voting/soft_voting.py](soft_voting/soft_voting.py): merge, metrics, confusion matrices
- [soft_voting/visualization_scripts.py](soft_voting/visualization_scripts.py): report plots generated from evaluation JSON

## Input Data Contract

### Soft voting input files

| File | Required columns | Description |
|---|---|---|
| `soft_voting/data/fe.csv` | `customer_unique_id`, `predicted_satisfaction` | FE probabilities |
| `soft_voting/data/llm.csv` | `customer_id`, `P_LLM` | LLM probabilities |
| `soft_voting/data/data_stock.csv` | `customer_unique_id`, `review_score` | Ground-truth labels |

Label mapping:

- `review_score <= 3` -> `0`
- `review_score >= 4` -> `1`

## Environment Setup

Requirements:

- Python 3.10+
- LM Studio (required for LLM module)

Install dependencies:

```bash
pip install -r requirement.txt
```

If `pandas` is missing in your current environment:

```bash
pip install pandas
```

## End-to-End Execution

Run commands from the repository root.

### Step 1: Features Engineering

```bash
python features_engineering/validation/predict_satisfaction.py
python features_engineering/validation/predict_acc.py
```

Note: [features_engineering/validation/predict_satisfaction.py](features_engineering/validation/predict_satisfaction.py) currently uses absolute paths such as `d:/Olist/...`. Update these paths for your local machine before execution.

### Step 2: LLM Extraction

Optional sampling/preprocessing:

```bash
python -m llm_extraction.processing
```

Run full Phase 1 + Phase 2 pipeline:

```bash
python -m llm_extraction.main
```

LM Studio notes:

- Default API endpoint: `http://localhost:1234/v1`
- Ensure models configured in [llm_extraction/config.py](llm_extraction/config.py) are loaded:
  - `LLM_MODEL_PHASE1`
  - `LLM_MODEL_PHASE2`

### Step 3: Soft Voting + Visualization

```bash
python soft_voting/soft_voting.py
python soft_voting/visualization_scripts.py
```

## Key Outputs

### Metrics JSON

- `soft_voting/data/evaluation_metrics.json`

### Ensemble visualizations

- `soft_voting/results/confusion_matrices_from_json.png`
- `soft_voting/results/core_metrics_comparison_overview.png`
- `soft_voting/results/rmse_comparison.png`
- `soft_voting/results/outcome_breakdown_stacked.png`

### Features Engineering validation plots

- `features_engineering/validation/confusion_matrix.png`
- `features_engineering/validation/roc_curve.png`
- `features_engineering/validation/precision_recall_curve.png`

## Evaluation Metrics

Tracked metrics include:

- Accuracy
- AUC
- F1-score
- RMSE
- Precision
- Recall
- Specificity
- Balanced Accuracy
- Confusion matrix counts: `tp`, `tn`, `fp`, `fn`

## Quick Troubleshooting

1. LM Studio connection errors
   - Confirm local server is running and configured model names are loaded.
2. Import/package errors
   - Verify the active Python environment and installed dependencies.
3. Low overlap during soft-voting merge
   - Check customer identifiers and data types across all three input files.
4. Features Engineering path errors
   - Replace hardcoded absolute paths with valid local paths.

## Future Improvements

- Centralize all file paths in shared configuration.
- Pin dependency versions for stronger reproducibility.
- Add automated tests for parser and metric components.
- Provide a single orchestration command to run the entire pipeline.

