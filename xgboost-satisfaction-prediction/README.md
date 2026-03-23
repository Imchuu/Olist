# XGBoost Customer Satisfaction Prediction

This project implements an XGBoost model to predict customer satisfaction scores based on processed features from the Olist e-commerce dataset. The model utilizes k-fold cross-validation to ensure robust performance and outputs predicted satisfaction scores on a scale of 0-1.

## Project Structure

```
xgboost-satisfaction-prediction
├── data
│   ├── olist_final_dataset_labeled.csv  # Final dataset with labeled customer satisfaction scores
│   ├── olist_orders_dataset_clean.csv    # Cleaned order data
│   ├── olist_order_items_dataset_clean.csv # Cleaned order item data
│   ├── olist_order_payments_dataset_clean.csv # Cleaned payment data
│   ├── olist_products_dataset_clean.csv   # Cleaned product data
│   ├── olist_order_reviews_dataset_clean.csv # Cleaned review data
│   └── olist_customers_dataset_clean.csv  # Cleaned customer data
├── scripts
│   └── predict_satisfaction.py             # Script to implement the XGBoost model
├── requirements.txt                        # List of dependencies
└── README.md                               # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd xgboost-satisfaction-prediction
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the model and generate predictions, execute the following command:
```
python scripts/predict_satisfaction.py
```

This will produce a CSV file containing customer IDs and their predicted satisfaction scores.

## Dataset Information

The dataset used for training and testing the model is `olist_final_dataset_labeled.csv`, which contains various features related to customer orders, products, and reviews, along with the corresponding satisfaction scores.

## Model Information

The model implemented in `predict_satisfaction.py` uses XGBoost with k-fold cross-validation to ensure that the predictions are reliable and generalizable. The output is a CSV file with customer IDs and their predicted satisfaction scores on a scale of 0-1.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.