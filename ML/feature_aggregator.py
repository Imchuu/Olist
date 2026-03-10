"""Aggregate order-level rows into customer-level features for Flow 2."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_to_customer_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order-level data into one row per unique customer.

    Groups by 'customer_unique_id' and computes the following features:

    - num_orders              : total number of orders
    - avg_price               : mean item price across all orders
    - avg_freight             : mean freight value across all orders
    - total_spend             : sum of (price + freight) across all orders
    - avg_delivery_time       : mean days from purchase to actual delivery
    - late_delivery_ratio     : fraction of orders delivered after estimated date
    - max_delay_days          : maximum days delayed (0 if never late)
    - num_product_categories  : count of distinct product categories
    - num_payment_types       : count of distinct payment methods used

    Args:
        df: Order-level DataFrame produced by load_dataset().

    Returns:
        Customer-level DataFrame with one row per customer_unique_id,
        plus a 'customer_id' alias column for pipeline compatibility.
    """
    logger.info("Aggregating %d order rows to customer level...", len(df))

    # ------------------------------------------------------------------ #
    # Delivery timing features                                            #
    # ------------------------------------------------------------------ #
    df = df.copy()

    if "order_delivered_customer_date" in df.columns and "order_purchase_timestamp" in df.columns:
        df["delivery_time_days"] = (
            df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 86_400
    else:
        df["delivery_time_days"] = float("nan")

    if "order_delivered_customer_date" in df.columns and "order_estimated_delivery_date" in df.columns:
        df["delay_days"] = (
            df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
        ).dt.total_seconds() / 86_400
        df["is_late"] = (df["delay_days"] > 0).astype(float)
    else:
        df["delay_days"] = float("nan")
        df["is_late"] = float("nan")

    df["total_order_value"] = df["price"].fillna(0) + df["freight_value"].fillna(0)

    # ------------------------------------------------------------------ #
    # Aggregation                                                         #
    # ------------------------------------------------------------------ #
    agg = df.groupby("customer_unique_id").agg(
        num_orders=("order_id", "nunique"),
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        total_spend=("total_order_value", "sum"),
        avg_delivery_time=("delivery_time_days", "mean"),
        late_delivery_ratio=("is_late", "mean"),
        max_delay_days=("delay_days", lambda s: s.clip(lower=0).max()),
        num_product_categories=("product_category_name_english", "nunique"),
        num_payment_types=("payment_type", "nunique"),
    ).reset_index()

    # Round floats for cleaner prompt text
    float_cols = [
        "avg_price", "avg_freight", "total_spend",
        "avg_delivery_time", "late_delivery_ratio", "max_delay_days",
    ]
    agg[float_cols] = agg[float_cols].round(2)
    agg.fillna(0, inplace=True)

    # Pipeline uses 'customer_id' as the output key
    agg["customer_id"] = agg["customer_unique_id"]

    logger.info("Aggregated to %d unique customers.", len(agg))
    return agg
