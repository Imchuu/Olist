import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import csv

# Load the dataset (updated path to original location)
data_path = 'd:/Olist/olist_final_dataset_labeled.csv'
df = pd.read_csv(data_path)

# Load orders to get customer_id (updated path)
orders_path = 'd:/Olist/olist_orders_dataset_clean.csv'
try:
    orders = pd.read_csv(orders_path)
    print(f"Orders file loaded successfully. Columns: {list(orders.columns)}")
except FileNotFoundError:
    raise FileNotFoundError(f"Orders file not found at {orders_path}. Please ensure the file exists in d:\\Olist\\.")

# Load customers to get customer_unique_id
customers_path = 'd:/Olist/olist_customers_dataset_clean.csv'
try:
    customers = pd.read_csv(customers_path)
    print(f"Customers file loaded successfully. Columns: {list(customers.columns)}")
except FileNotFoundError:
    raise FileNotFoundError(f"Customers file not found at {customers_path}. Please ensure the file exists in d:\\Olist\\.")

# Check if required columns exist in orders and customers
if 'order_id' not in orders.columns or 'customer_id' not in orders.columns:
    print(f"Available columns in orders: {list(orders.columns)}")
    print("Warning: Required columns not found in orders. Proceeding without merge (using placeholder).")
    df['customer_unique_id'] = 'unknown'  # Placeholder if merge fails
elif 'customer_id' not in customers.columns or 'customer_unique_id' not in customers.columns:
    print(f"Available columns in customers: {list(customers.columns)}")
    print("Warning: Required columns not found in customers. Proceeding without merge (using placeholder).")
    df['customer_unique_id'] = 'unknown'  # Placeholder if merge fails
else:
    # Merge to add customer_id from orders
    df = df.merge(orders[['order_id', 'customer_id']], on='order_id', how='left')
    # Merge to add customer_unique_id from customers
    df = df.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='left')

# Check for required columns
required_cols = ['review_binary'] + [col for col in df.columns if col.startswith('norm_')]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in dataset: {missing_cols}")

# Prepare features and target variable
feature_cols = [col for col in df.columns if col.startswith('norm_')]
X = df[feature_cols]
y = df['review_binary']

# Initialize XGBoost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = np.zeros(X.shape[0])

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]  # Get probability of class 1
    predictions[test_index] = preds

# Prepare output DataFrame
output_df = pd.DataFrame({
    'customer_unique_id': df['customer_unique_id'],
    'predicted_satisfaction': predictions
})

# Save to CSV (updated path)
output_file_path = 'd:/Olist/predicted_satisfaction_scores.csv'
output_df.to_csv(output_file_path, index=False)

print(f'Predicted satisfaction scores saved to {output_file_path}')