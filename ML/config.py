"""Configuration constants for Flow 2 LLM customer satisfaction prediction."""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
INPUT_PATH = Path("ML/data/dataset.csv")
OUTPUT_PATH = Path("ML/results/llm_predictions.csv")
PHASE1_OUTPUT_PATH = Path("ML/results/phase1_narratives.csv")

PROCESSING_INPUT_PATH = Path("data/olist_full_merged.csv")
PROCESSING_OUTPUT_PATH = Path("data/olist_sampled_10k.csv")
PROCESSING_TARGET_ROWS = 10_000
SAMPLING_SEED_PRIMARY = 42
SAMPLING_SEED_ADJUSTMENT = 0

# -----------------------------------------------------------------------------
# Local LLM API (LM Studio - OpenAI compatible)
# -----------------------------------------------------------------------------
LOCAL_API_BASE_URL = "http://localhost:1234/v1"

# User-requested explicit phase model names.
LLM_MODEL_PHASE1 = "qwen2.5-7b-instruct"
LLM_MODEL_PHASE2 = "deepseek-r1-distill-qwen-14b"

# Default model used when call site does not specify phase explicitly.
LLM_MODEL = LLM_MODEL_PHASE2

LLM_TEMPERATURE = 0.0
PHASE1_MAX_TOKENS = 120
PHASE2_MAX_TOKENS = 24
REQUEST_TIMEOUT = 120
CONNECTION_CHECK_TIMEOUT = 5
CHAT_ENDPOINT = "/chat/completions"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0
PROGRESS_LOG_EVERY_N = 50

# Set to 0 to process all customers. Use a small value for speed benchmarking.
MAX_CUSTOMERS = 0

# Number of worker threads for concurrent LLM requests. Set 1 to disable multithreading.
MAX_WORKERS = 10

# -----------------------------------------------------------------------------
# Data schema and parsing
# -----------------------------------------------------------------------------
REQUIRED_DATASET_COLUMNS = [
    "customer_unique_id",
    "order_id",
    "review_score",
    "price",
    "freight_value",
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "payment_type",
    "product_category_name_english",
]

DATETIME_COLUMNS = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "order_approved_at",
    "order_delivered_carrier_date",
]

# -----------------------------------------------------------------------------
# Pipeline behavior
# -----------------------------------------------------------------------------
STOP_AFTER_PHASE1 = True
FAIL_FAST_ON_LLM_ERROR = True
