# predict_satisfaction.py

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# Load the datasets
olist_data = pd.read_csv('olist_final_dataset_labeled.csv')
predicted_data = pd.read_csv('predicted_satisfaction_scores.csv')

# Merge the datasets on customer_unique_id
merged_data = pd.merge(olist_data, predicted_data, on='customer_unique_id', how='inner')

# Create true labels: 0 for dissatisfied (review_score 1-3), 1 for satisfied (review_score 4-5)
merged_data['true_label'] = merged_data['review_score'].apply(lambda x: 0 if x <= 3 else 1)

# Predicted labels for accuracy: threshold at 0.5
merged_data['predicted_label'] = merged_data['predicted_satisfaction'].apply(lambda x: 1 if x >= 0.5 else 0)

# Calculate accuracy
accuracy = accuracy_score(merged_data['true_label'], merged_data['predicted_label'])

# Calculate AUC using predicted_satisfaction as probabilities
auc = roc_auc_score(merged_data['true_label'], merged_data['predicted_satisfaction'])

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")